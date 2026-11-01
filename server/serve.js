// GF2 RAG 检索服务：dense ∪ BM25 → RRF → 行去重 → rerank → JSON
// 管线照 shujuku 交火模式：宽进候选、RRF 融合、rerank 失败回退 embedding 序
const http=require('http'),fs=require('fs'),path=require('path');
const PORT=+(process.env.PORT||5240);
const DIR=__dirname+'/'+(process.env.IDX||'index');
const CFG_FILE=path.join(__dirname,'config.json');

/* ---------- 可持久化动态配置 ---------- */
const defaultCfg={
  chatBase:'https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1',
  chatKey:'''-XVwli1ZbSNCJ-TCopdAiBRx9OHLabI1oNZFpQnNvGvVmN4u5irJeFlq8QafHHbxA',
  chatModel:process.env.CHAT_MODEL||'deepseek-v4.1-flash-1017',
  embedBase:'https://api.siliconflow.cn/v1',
  embedKey:'''',
  embedModel:process.env.EMBED_MODEL||'Qwen/Qwen3-Embedding-8B',
  rerankModel:process.env.RERANK_MODEL||'Qwen/Qwen3-Reranker-8B'
};
let runtimeCfg={...defaultCfg};
if(fs.existsSync(CFG_FILE)){
  try{runtimeCfg={...defaultCfg,...JSON.parse(fs.readFileSync(CFG_FILE,'utf8'))}}catch(e){}
}else{
  try{fs.writeFileSync(CFG_FILE,JSON.stringify(runtimeCfg,null,2))}catch(e){}
}
function saveCfg(patch){
  for(const k of Object.keys(defaultCfg)){
    if(patch&&typeof patch[k]==='string'&&patch[k].trim())runtimeCfg[k]=patch[k].trim();
  }
  try{fs.writeFileSync(CFG_FILE,JSON.stringify(runtimeCfg,null,2))}catch(e){}
}

/* ---------- 加载索引 ---------- */
const chunks=JSON.parse(fs.readFileSync(DIR+'/meta.json','utf8'));
const raw=fs.readFileSync(DIR+'/vectors.f32');
const vectors=new Map();let dim=0;
for(let off=0;off+8<=raw.length;){
  const idx=raw.readUInt32LE(off),n=raw.readUInt32LE(off+4);off+=8;
  const v=new Float32Array(n);
  for(let i=0;i<n;i++)v[i]=raw.readFloatLE(off+i*4);
  off+=n*4;vectors.set(idx,v);dim=n;
}
for(const v of vectors.values()){let s=0;for(const x of v)s+=x*x;s=Math.sqrt(s)||1;for(let i=0;i<v.length;i++)v[i]/=s}
console.log('index loaded:',vectors.size,'vectors, dim',dim,'chunks:',chunks.length);

/* ---------- BM25（抄 shujuku：CJK 单字+bigram） ---------- */
const K1=1.5,B=0.75;
function tokens(text){
  const s=String(text||'').toLowerCase(),t=[];
  const re=/[a-z0-9_]+|[㐀-鿿]+/gi;let m;
  while((m=re.exec(s))){
    const p=m[0];
    if(/^[㐀-鿿]+$/u.test(p)){for(const c of p)t.push(c);for(let i=0;i<p.length-1;i++)t.push(p.slice(i,i+2))}
    else t.push(p);
  }
  return t;
}
const corpus={docs:[],df:new Map(),avg:0};
let total=0;
for(let i=0;i<chunks.length;i++){
  const c=chunks[i];
  const tk=tokens(c.tag+' '+c.characters+' '+c.text);
  const freq=new Map();
  for(const t of tk)freq.set(t,(freq.get(t)||0)+1);
  for(const t of freq.keys())corpus.df.set(t,(corpus.df.get(t)||0)+1);
  corpus.docs.push({i,freq,len:tk.length});total+=tk.length;
}
corpus.avg=total/corpus.docs.length;
corpus.N=corpus.docs.length;
console.log('bm25 corpus:',corpus.N,'docs, avg len',corpus.avg|0);
function bm25(qt){
  const out=[];
  for(const d of corpus.docs){
    let s=0;
    for(const t of qt){
      const tf=d.freq.get(t)||0;if(!tf)continue;
      const df=corpus.df.get(t)||0;
      const idf=Math.log(1+(corpus.N-df+0.5)/(df+0.5));
      s+=idf*(tf*(K1+1))/(tf+K1*(1-B+B*d.len/corpus.avg));
    }
    if(s>0)out.push({i:d.i,score:s});
  }
  return out.sort((a,b)=>b.score-a.score);
}

/* ---------- 关键词 AI（照 shujuku 交火模式 + 29章全目录 + 49条角色目录） ---------- */
function buildCatalog(){
  const rd=n=>{for(const p of [path.join(__dirname,n),'D:/gf2-work/'+n]){try{return fs.readFileSync(p,'utf8')}catch(e){}}return''};
  const lines=['【《少女前线2：追放》主线1-28章速查目录】'];
  try{
    const chs=rd('chapters.jsonl').trim().split('\n').map(JSON.parse).sort((a,b)=>a.chapter-b.chapter);
    for(const x of chs){
      const ch=x.chapter===100?'番外':('第'+(x.chapter-100)+'章');
      lines.push(`${ch}《${x.title}》：${x.description}（触发/人物/设定：${x.triggerWhen||''}；${(x.characters||[]).join('/')}；${(x.factions||[]).join('/')}）`);
    }
  }catch(e){}
  lines.push('\n【角色个人/密会剧情速查目录】');
  try{
    const cs=rd('entries-char.jsonl').trim().split('\n').map(JSON.parse);
    for(const x of cs){
      lines.push(`[${x.protagonist}]《${x.title}》：${x.description}（${x.triggerWhen}）`);
    }
  }catch(e){}
  return lines.join('\n');
}
const GF2_CATALOG=buildCatalog();
console.log('catalog loaded, chars:',GF2_CATALOG.length);

const KW_SYS=`你是《少女前线2：追放》剧情检索守门员与交火关键词生成助手。
主模型知识截止于2025年3月（缺乏第15-28章后期主线及角色个人剧情的原文细节）。
下方为你配备了完整的《主线1-28章与角色剧情速查目录》，请对照目录理解用户提到的暗语、代称或事件（例如“粉毛幼女/培养罐”=O-201/海伦娜实验体/星渊；“两个闪电”=朱诺/OTs-14/焚化炉；“那个芯片病毒”=克拉德涅茨/新威尼斯/AK12等）。

${GF2_CATALOG}

【任务判断】
1. 如果当前用户输入只是纯粹的日常寒暄、动作描写，且完全不涉及需要查阅过往剧情/设定/人物经历/具体台词细节，请直接在 <keywords> 中输出 NONE。
2. 如果涉及任何具体剧情回顾、特定事件、专有装置/组织、人物过往创伤或名场面台词，请对照上方目录，把模糊指代转化为精确的专有名词、人名、地点、道具名（注意：不要输出“第X章《xxx》”这样的章节标题，只输出剧本原文里会出现的实体词、人名、装置名和动作词），输出 6~14 个最相关的检索关键词。

【输出格式 — 必须严格遵守】
<thinking>
用一两句话对照目录判断：是否需要检索原文？如果模糊代指，对应目录里的哪个章节/人物/专有名词？
</thinking>
<keywords>关键词1，关键词2，关键词3</keywords>（若无需检索则写 <keywords>NONE</keywords>）`;

async function genKeywords(input,context,cfg){
  const base=(cfg.chatBase||runtimeCfg.chatBase).replace(/\/+$/,'');
  const key=cfg.chatKey||runtimeCfg.chatKey;
  const model=cfg.chatModel||runtimeCfg.chatModel;
  const user='最近上下文：\n'+(context||'（无）')+'\n\n当前用户输入：\n'+input+'\n\n请判断是否需要检索并输出 <thinking> 与 <keywords>。';
  const r=await fetch(base+'/chat/completions',{method:'POST',headers:{'Authorization':'Bearer '+key,'Content-Type':'application/json'},body:JSON.stringify({model,messages:[{role:'system',content:KW_SYS},{role:'user',content:user}],temperature:0.1,max_tokens:450})});
  if(!r.ok)throw new Error('kw '+r.status);
  const t=(await r.json()).choices[0].message.content;
  const m=t.match(/<keywords>([\s\S]*?)<\/keywords>/i)||t.match(/关键词[：:]([\s\S]+)$/);
  if(!m)return {skip:false,kws:[]};
  const raw=m[1].trim();
  if(/^none$/i.test(raw))return {skip:true,kws:[]};
  return {skip:false,kws:raw.split(/[，,、]/).map(x=>x.trim()).filter(Boolean)};
}

/* ---------- dense ---------- */
async function embedQuery(q,cfg){
  const base=(cfg.embedBase||runtimeCfg.embedBase).replace(/\/+$/,'');
  const key=cfg.embedKey||runtimeCfg.embedKey;
  const model=cfg.embedModel||runtimeCfg.embedModel;
  const r=await fetch(base+'/embeddings',{method:'POST',headers:{'Authorization':'Bearer '+key,'Content-Type':'application/json'},body:JSON.stringify({model,input:[q],encoding_format:'float'})});
  if(!r.ok)throw new Error('embed '+r.status);
  const v=(await r.json()).data[0].embedding;
  let s=0;for(const x of v)s+=x*x;s=Math.sqrt(s)||1;
  return v.map(x=>x/s);
}
function denseTop(qv,n){
  const out=[];
  for(const [i,v] of vectors){
    let s=0;for(let k=0;k<dim;k++)s+=qv[k]*v[k];
    out.push({i,score:s});
  }
  out.sort((a,b)=>b.score-a.score);
  return out.slice(0,n);
}

/* ---------- RRF ---------- */
function rrf(lists,k,limit){
  const m=new Map();
  for(const list of lists)list.forEach((c,idx)=>{
    const p=m.get(c.i)||{i:c.i,score:0};
    p.score+=1/(k+idx+1);m.set(c.i,p);
  });
  return [...m.values()].sort((a,b)=>b.score-a.score).slice(0,limit);
}

/* ---------- rerank ---------- */
async function rerank(query,docs,retry,cfg){
  const base=(cfg.embedBase||runtimeCfg.embedBase).replace(/\/+$/,'');
  const key=cfg.embedKey||runtimeCfg.embedKey;
  const model=cfg.rerankModel||runtimeCfg.rerankModel;
  const r=await fetch(base+'/rerank',{method:'POST',headers:{'Authorization':'Bearer '+key,'Content-Type':'application/json'},body:JSON.stringify({model,query,documents:docs,top_n:docs.length,return_documents:false})});
  if(!r.ok){
    if(retry<2){await new Promise(x=>setTimeout(x,2000));return rerank(query,docs,retry+1,cfg)}
    throw new Error('rerank '+r.status);
  }
  const d=await r.json();
  return d.results.map(x=>({index:x.index,score:x.relevance_score}));
}

/* ---------- HTTP ---------- */
http.createServer(async(req,res)=>{
  const send=(o,c=200)=>{res.writeHead(c,{'Content-Type':'application/json','Access-Control-Allow-Origin':'*','Access-Control-Allow-Headers':'Content-Type, Authorization','Access-Control-Allow-Methods':'GET, POST, OPTIONS'});res.end(JSON.stringify(o))};
  if(req.method==='OPTIONS'){res.writeHead(204,{'Access-Control-Allow-Origin':'*','Access-Control-Allow-Headers':'Content-Type, Authorization','Access-Control-Allow-Methods':'GET, POST, OPTIONS'});return res.end()}
  if(req.url==='/health')return send({ok:true,chunks:chunks.length,vectors:vectors.size,config:runtimeCfg});
  if(req.method==='GET'&&req.url.startsWith('/context')){
    try{
      const u=new URL(req.url,'http://x');
      const id=u.searchParams.get('id'),before=+(u.searchParams.get('before')||3),after=+(u.searchParams.get('after')||3);
      const pos=chunks.findIndex(c=>c.id===id);
      if(pos<0)return send({error:'chunk not found'},404);
      const g=String(chunks[pos].group||'');
      const items=[];
      for(let j=pos-before;j<=pos+after;j++){
        if(j<0||j>=chunks.length)continue;
        if(String(chunks[j].group||'')!==g)continue;
        items.push({id:chunks[j].id,group:chunks[j].group,tag:chunks[j].tag,anchor:j===pos,text:chunks[j].text});
      }
      return send({id,group:g,text:items.map(x=>x.text).join('\n'),items});
    }catch(e){return send({error:e.message},500)}
  }
  if(req.method==='GET'&&req.url==='/config')return send(runtimeCfg);
  if(req.method==='POST'&&req.url==='/config'){
    let body='';for await(const c of req)body+=c;
    try{saveCfg(JSON.parse(body||'{}'));return send({ok:true,config:runtimeCfg})}
    catch(e){return send({error:e.message},400)}
  }
  if(req.method==='POST'&&req.url==='/models'){
    let body='';for await(const c of req)body+=c;
    try{
      const {baseUrl,apiKey}=JSON.parse(body||'{}');
      const b=(baseUrl||runtimeCfg.chatBase).replace(/\/+$/,'');
      const k=apiKey||runtimeCfg.chatKey;
      const r=await fetch(b+'/models',{headers:{'Authorization':'Bearer '+k}});
      if(!r.ok)throw new Error('HTTP '+r.status);
      const j=await r.json();
      return send({ok:true,models:(j.data||[]).map(x=>x.id)});
    }catch(e){return send({ok:false,error:e.message},400)}
  }
  if(req.method==='POST'&&req.url==='/search'){
    let body='';for await(const c of req)body+=c;
    try{
      const payload=JSON.parse(body);
      const {query,k=8,denseN=50,sparseN=50,useRerank=true,filter,augment=false,context='',apiConfig={},keywords=[],expand=1}=payload;
      if(!query)return send({error:'no query'},400);
      if(apiConfig&&Object.keys(apiConfig).length)saveCfg(apiConfig);
      const t0=Date.now();
      let effQuery=query,kws=Array.isArray(keywords)?keywords.map(x=>String(x).trim()).filter(Boolean):[],kwError=null;
      if(!kws.length&&augment){
        try{
          const gk=await genKeywords(query,context,apiConfig);
          if(gk.skip)return send({took_ms:Date.now()-t0,rerank:'skipped_by_gatekeeper',keywords:[],results:[]});
          kws=gk.kws;
        }catch(e){kwError=e.message}
      }
      if(kws.length)effQuery=query+'\n关键词：'+kws.join('，');
      const qt=tokens(effQuery+' '+kws.join(' '));
      const sparse=bm25(qt).slice(0,sparseN);
      const qv=await embedQuery(effQuery,apiConfig);
      const dense=denseTop(qv,denseN);
      let fused=rrf([dense,sparse],60,Math.max(denseN,sparseN));
      if(filter&&filter.group)fused=fused.filter(c=>chunks[c.i].group===filter.group);
      const seen=new Set();const dedup=[];
      for(const c of fused){const g=String(chunks[c.i].group||'').replace(/^CN_/,'');if(seen.has(g))continue;seen.add(g);dedup.push(c)}
      let ranked=dedup,rerankStatus='skipped';
      if(useRerank&&dedup.length>Math.max(k,2)){
        try{
          const docs=dedup.map(c=>chunks[c.i].tag+'\n'+chunks[c.i].text.slice(0,800));
          const rs=await rerank(effQuery,docs,0,apiConfig);
          ranked=rs.map(r=>({...dedup[r.index],rerankScore:r.score}));
          rerankStatus='applied';
        }catch(e){rerankStatus='failed:'+e.message.slice(0,80);ranked=dedup}
      }
      send({
        debug:{sparseTop:sparse.slice(0,5).map(c=>({id:chunks[c.i].id,score:+c.score.toFixed(2)})),denseTop:dense.slice(0,5).map(c=>({id:chunks[c.i].id,score:+c.score.toFixed(3)})),kwError},
        took_ms:Date.now()-t0,rerank:rerankStatus,keywords:kws,
        results:ranked.slice(0,k).map(c=>{
          const ch=chunks[c.i];
          let text=ch.text;
          if(expand>0){
            const g=String(ch.group||'');
            const merge=(j)=>j>=0&&j<chunks.length&&String(chunks[j].group||'')===g?chunks[j].text:null;
            const prev=[],next=[];
            for(let d=1;d<=expand;d++){
              const p=merge(c.i-d);if(p)prev.unshift(p);
              const nx=merge(c.i+d);if(nx)next.push(nx);
            }
            if(prev.length||next.length)text=[...prev,text,...next].join('\n');
          }
          return{
            id:ch.id,group:ch.group,tag:ch.tag,
            score:+(c.rerankScore??c.score).toFixed(4),
            text
          };
        })
      });
    }catch(e){send({error:e.message},500)}
    return;
  }
  send({error:'not found'},404);
}).listen(PORT,()=>console.log('gf2-rag on :'+PORT));
