"""Real Chromium UI tests using in-memory module transport.
URL navigation is blocked by administrator policy on the validation host.
App / configuration / renderer modules are unmodified except for import transport.
No renderer, callbacks or state store are mocked. This opaque-origin DOM has no
navigator.gpu, so this test does NOT claim a browser/WebGPU rendering playthrough.
"""
from pathlib import Path
import os,re,posixpath,json,base64
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];DOCS=ROOT/'docs';DOCS.mkdir(exist_ok=True)
checks=[];errors=[]
def check(name,value):
 checks.append({'name':name,'passed':bool(value)});print(('PASS ' if value else 'FAIL ')+name,flush=True)
 if not value:raise AssertionError(name)
def asset(path):
 from urllib.parse import urlparse
 target=(ROOT/urlparse(path).path.lstrip('./')).resolve()
 if not target.is_relative_to(ROOT) or not target.is_file():return {'ok':False,'data':''}
 return {'ok':True,'data':base64.b64encode(target.read_bytes()).decode()}
def rewrite(path,text):
 def sub(m):
  spec=m.group(2)
  if not spec.startswith('.'):return m.group(0)
  dest=posixpath.normpath(posixpath.join(path.parent.relative_to(ROOT).as_posix(),spec))
  return m.group(1)+'dermis/'+dest+m.group(3)
 return re.sub(r'((?:from\s*|import\s*\(?\s*)[\'"])([^\'"]+)([\'"])',sub,text)
modules={}
for path in list(ROOT.glob('src/**/*.js'))+list(ROOT.glob('vendor/**/*.js')):
 modules['dermis/'+path.relative_to(ROOT).as_posix()]='data:text/javascript;base64,'+base64.b64encode(rewrite(path,path.read_text()).encode()).decode()
html=ROOT.joinpath('index.html').read_text();html=re.sub(r'<script type="module".*?</script>','',html,flags=re.S);html=html.replace('<link rel="stylesheet" href="src/style.css">','<style>'+ROOT.joinpath('src/style.css').read_text()+'</style>')
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH','/usr/bin/chromium'),headless=True,args=['--no-sandbox'])
  page=browser.new_page(viewport={'width':1440,'height':960});page.on('pageerror',lambda e:errors.append(str(e)));page.expose_function('__fixtureAsset',asset);page.set_content(html)
  page.evaluate('''()=>{const native=window.fetch;window.fetch=async path=>{if(String(path).startsWith('blob:')||String(path).startsWith('data:'))return native(path);const a=await __fixtureAsset(String(path));return new Response(Uint8Array.from(atob(a.data),x=>x.charCodeAt(0)),{status:a.ok?200:404});};window.downloads=[];HTMLAnchorElement.prototype.click=function(){if(this.download)downloads.push({name:this.download,url:this.href});};}''')
  page.evaluate('''imports=>{const map=document.createElement('script');map.type='importmap';map.textContent=JSON.stringify({imports});document.head.appendChild(map);const entry=document.createElement('script');entry.type='module';entry.textContent="import 'dermis/src/app.js'";document.body.appendChild(entry);}''',modules)
  page.wait_for_function('!!window.dermisApp',timeout=30000)
  check('Complete application UI imports in Chromium',page.locator('#initialize').is_visible())
  check('All 19 parameter sliders are present',page.locator('input[type=range]').count()==19)
  page.locator('#melanin').fill('0.61');page.locator('#melanin').dispatch_event('input');check('Skin slider updates validated state',page.evaluate('dermisApp.config.melanin')==.61)
  check('Slider value is reflected in its output label',page.locator('#melanin-value').text_content()=='0.61')
  page.locator('#iris').select_option('3');check('Iris colour selection updates state',page.evaluate('dermisApp.config.iris')==3)
  page.locator('#hair').uncheck();check('Hair visibility is a real boolean setting',page.evaluate('dermisApp.config.hair') is False)
  page.locator('#quality').select_option('high');check('Pre-initialization quality choice is retained',page.evaluate('dermisApp.config.quality')=='high')
  page.locator('[data-camera=threequarter]').click();check('Camera preset sets a three-quarter orbit',page.evaluate('dermisApp.config.orbitYaw')==.63)
  page.locator('[data-view="2"]').dispatch_event('click');check('Normal-inspection button sets the render mode',page.evaluate('dermisApp.config.view')==2)
  page.locator('#animate').click();check('Living-portrait toggle changes animation state',page.evaluate('dermisApp.config.animation') is True)
  page.locator('[data-look=dark]').click();check('Starting look sets skin and eye pigments together',page.evaluate('dermisApp.config.melanin===.76&&dermisApp.config.iris===3'))
  page.locator('#save-button').click();saved=page.evaluate('async()=>{const x=downloads.at(-1);return {name:x.name,text:await(await fetch(x.url)).text()}}')
  check('Save look generates a portable versioned JSON file',json.loads(saved['text'])['format']=='dermis-look')
  f=DOCS/'ui-saved-look.json';f.write_text(saved['text'])
  page.locator('#reset-look').click();page.locator('#load-file').set_input_files(str(f));page.wait_for_timeout(200)
  check('Saved look reopens via browser file input',page.evaluate('dermisApp.config.melanin')==.76)
  bad=DOCS/'invalid-look-test.json';bad.write_text(json.dumps({'format':'dermis-look','version':1,'config':{'melanin':'bad'}}));page.locator('#load-file').set_input_files(str(bad));page.wait_for_timeout(150)
  check('Malformed look is rejected without altering the previous look',page.evaluate('dermisApp.config.melanin')==.76);bad.unlink()
  page.locator('#source-button').click();page.wait_for_function('document.querySelector("#source-code").textContent.includes("generateSurface")')
  check('Source dialog displays the actual geometry CUDA file',page.locator('#source-dialog').is_visible())
  page.locator('[data-source=appearance]').click();check('Source tab exposes actual skin/eye/hair shading code','shadePortrait' in page.locator('#source-code').text_content())
  page.locator('#download-source').click();check('Source export uses a .cu filename',page.evaluate('downloads.at(-1).name')=='appearance.cu')
  page.locator('#close-source').click();page.locator('#reset-look').click();page.screenshot(path=str(DOCS/'interface-dom.png'))
  check('Desktop layout fits the viewport',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
  page.set_viewport_size({'width':390,'height':844});page.locator('#mobile-controls').click();check('Mobile controls drawer opens',page.locator('#inspector').is_visible())
  check('Mobile layout has no horizontal document overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
  page.screenshot(path=str(DOCS/'interface-mobile.png'));page.locator('#mobile-controls').click()
  page.locator('#initialize').click();page.wait_for_function('!document.querySelector("#error").hidden')
  check('Unavailable WebGPU is reported rather than a fake CPU/image fallback','WebGPU' in page.locator('#error').text_content())
  check('PNG capture remains disabled without a real rendered framebuffer',page.locator('#capture-button').is_disabled())
  check('No uncaught application JavaScript errors',not errors)
  browser.close()
finally:
 (DOCS/'browser-validation.json').write_text(json.dumps({'passed':sum(x['passed']for x in checks),'total':len(checks),'checks':checks,'errors':errors,'backend':'Chromium real DOM with in-memory module/fetch transport; no WebGPU context or renderer mock','limitation':'Direct HTTP navigation was blocked by administrator policy. End-to-end browser WebGPU presentation is unverified.'},indent=2))
