// chunks.jsonl → index/（meta.json + vectors.f32）
// 批量调 SiliconFlow embedding，断点续跑
const fs=require('fs'),path=require('path');
const KEY='[REDACTED]';
const BASE='https://api.siliconflow.cn/v1';
const MODEL=process.env.EMBED_MODEL||'Qwen/Qwen3-Embedding-8B';
const DIR=__dirname+'/'+(process.env.IDX||'index');
const CHUNKS=process.env.CHUNKS||'D:/gf2-work/chunks.jsonl';
const BATCH=+(process.env.BATCH||32);
const CONC=+(process.env.CONC||4);
const MAXCH=+(process.env.MAX_CHARS||1800);

const OFF=+(process.env.OFF||0);const chunks=fs.readFileSync(CHUNKS,'utf8').split('\n').filter(Boolean).map(JSON.parse);
console.log('chunks:',chunks.length);
fs.mkdirSync(DIR,{recursive:true});
const vecPath=DIR+'/vectors.f32',donePath=DIR+'/done.json';
let dim=0, doneSet=new Set();
const metaPath=DIR+'/meta.json';
if(!fs.existsSync(metaPath)) fs.writeFileSync(metaPath,JSON.stringify(chunks));
if(fs.existsSync(donePath)){
  const d=JSON.parse(fs.readFileSync(donePath,'utf8'));
  doneSet=new Set(d.done); dim=d.dim;
  console.log('resume:',doneSet.size,'done, dim',dim);
}
const fd=fs.openSync(vecPath,'a');

async function embed(texts,retry){
  const r=await fetch(BASE+'/embeddings',{
    method:'POST',
    headers:{'Authorization':'Bearer '+KEY,'Content-Type':'application/json'},
    body:JSON.stringify({model:MODEL,input:texts,encoding_format:'float'})
  });
  if(!r.ok){
    const t=await r.text();
    if((r.status===429||r.status>=500)&&retry<5){await new Promise(x=>setTimeout(x,2000*(retry+1)));return embed(texts,retry+1)}
    throw new Error(r.status+' '+t.slice(0,300));
  }
  const d=await r.json();
  return d.data.map(x=>x.embedding);
}

(async()=>{
  const todo=[];
  for(let i=0;i<chunks.length;i++) if(!doneSet.has(i+OFF)) todo.push(i+OFF);
  console.log('todo:',todo.length);
  const batches=[];
  for(let i=0;i<todo.length;i+=BATCH)batches.push(todo.slice(i,i+BATCH));
  let done=0;
  async function worker(){
    while(batches.length){
      const b=batches.shift();if(!b)break;
      const texts=b.map(i=>(chunks[i-OFF].tag?chunks[i-OFF].tag+'\n':'')+chunks[i-OFF].text.slice(0,MAXCH));
      try{
        const vecs=await embed(texts,0);
        for(let j=0;j<b.length;j++){
          const idx=b[j],v=vecs[j];
          if(!dim)dim=v.length;
          const buf=Buffer.alloc(8+v.length*4);
          buf.writeUInt32LE(idx,0);buf.writeUInt32LE(v.length,4);
          for(let k=0;k<v.length;k++)buf.writeFloatLE(v[k],8+k*4);
          fs.writeSync(fd,buf);
          doneSet.add(idx);
        }
        done+=b.length;
        if(done%320<b.length)console.log('embedded',done,'/',todo.length,'dim',dim);
      }catch(e){console.log('FAIL batch@'+b[0],e.message.slice(0,120));await new Promise(x=>setTimeout(x,3000))}
    }
  }
  await Promise.all(Array.from({length:CONC},worker));
  fs.writeFileSync(donePath,JSON.stringify({dim,done:[...doneSet]}));
  fs.closeSync(fd);
  console.log('done. dim:',dim,'total:',doneSet.size);
})();
