export class StressRun {
 constructor(seconds,now=performance.now()){
  if(!Number.isFinite(seconds)||seconds<=0)throw Error('Invalid stress-test duration.');
  this.seconds=seconds;this.started=now;this.times=[];this.ended=null;this.reason='running';
 }
 record(duration,now=performance.now()){
  if(this.ended!==null)return false;
  this.times.push(duration);
  if(now-this.started>=this.seconds*1000)this.stop('completed',now);
  return this.ended!==null;
 }
 stop(reason='stopped',now=performance.now()){if(this.ended===null){this.ended=now;this.reason=reason;}}
 summary(now=performance.now()){
  const elapsed=Math.max(0,((this.ended??now)-this.started)/1000),sorted=[...this.times].sort((a,b)=>a-b),frames=sorted.length;
  return {reason:this.reason,requestedSeconds:this.seconds,elapsedSeconds:elapsed,frames,
   fps:elapsed>0?frames/elapsed:0,meanRenderMs:frames?this.times.reduce((a,b)=>a+b,0)/frames:0,
   p95RenderMs:frames?sorted[Math.ceil(frames*.95)-1]:0};
 }
}
