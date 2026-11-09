import './style.css';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

const $=id=>document.getElementById(id), BASE=import.meta.env.BASE_URL;
const localPage=['127.0.0.1','localhost','[::1]'].includes(location.hostname);
let apiOrigin=localPage?location.origin:'http://127.0.0.1:8765';
try{apiOrigin=localStorage.getItem('qamg-api')||apiOrigin;}catch{}
$('backend-url').value=apiOrigin;
let apiAvailable=false,modelReady=false,busy=false,mode='image',family='drive';
let structure,model,currentRun,currentSummary,animated=false,disassembling=false,animationStart=0,loadingVersion=0,previewURL;
const meshes=new Map(),selected=new Set(),staticRuns=new Map();
const viewport=$('viewport'),scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(38,1,.001,100);
camera.up.set(0,0,1);
let renderer,controls,floorGrid;

function status(message,error=false){$('job-status').textContent=message;$('job-status').classList.toggle('error',error);}
function setMode(value){mode=value;$('image-controls').hidden=value!=='image';$('param-controls').hidden=value!=='param';$('generate-label').textContent=value==='image'?'从图片生成 3D':'生成机械装配';$('encoding').disabled=value==='image';if(value==='image')$('encoding').value='compact_binary';document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===value));}
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));
function apiURL(path){return apiOrigin.replace(/\/$/,'')+path;}
async function json(url,options={}){
  const headers={...(options.body&&!(options.body instanceof FormData)?{'Content-Type':'application/json'}:{}),...(options.headers||{})};
  const response=await fetch(url.startsWith('/api/')?apiURL(url):url,{...options,headers});
  const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||data));return data;
}
function runBase(id){return staticRuns.has(id)&&!apiAvailable?`${BASE}gallery/${encodeURIComponent(id)}/`:apiURL(`/api/runs/${encodeURIComponent(id)}/`);}
function chooseFamily(value){family=value;document.querySelectorAll('[data-family]').forEach(b=>b.classList.toggle('active',b.dataset.family===value));$('size').min=value==='drive'?70:15;$('size').max=value==='drive'?150:50;$('intent').placeholder=value==='drive'?'例如：重载齿轮装配，轴长 110 mm':'例如：轻量夹持机构，开口 35 mm';}
document.querySelectorAll('[data-family]').forEach(b=>b.addEventListener('click',()=>{chooseFamily(b.dataset.family);$('size').value=family==='drive'?100:30;$('intent').value='';}));
function updateSelection(){
  for(const[id,object]of meshes)object.traverse(node=>{if(node.isMesh)for(const material of Array.isArray(node.material)?node.material:[node.material])if(material.emissive)material.emissive.setHex(selected.has(id)?0x184e37:0x000000);});
  document.querySelectorAll('[data-part]').forEach(b=>b.classList.toggle('active',selected.has(b.dataset.part)));
  document.querySelectorAll('[data-part-check]').forEach(b=>b.checked=selected.has(b.dataset.partCheck));
  $('merge-parts').disabled=selected.size<2||busy;$('selection-count').textContent=selected.size?`已选 ${selected.size} 个分件`:'在零件树勾选，或按 Shift 多选';
  const parts=structure?.parts.filter(p=>selected.has(p.id))||[];$('selection-label').hidden=!parts.length;
  if(parts.length)$('selection-label').textContent=`${parts.length===1?parts[0].label:parts.length+' 个分件'} · ${(parts.reduce((s,p)=>s+p.volume_mm3,0)/1000).toFixed(2)} cm³`;
}
function selectPart(id,toggle=false){if(!toggle)selected.clear();if(toggle&&selected.has(id))selected.delete(id);else selected.add(id);updateSelection();}
function setExplosion(value){if(!structure)return;for(const part of structure.parts){const object=meshes.get(part.id);if(object)object.position.copy(object.userData.originalPosition).addScaledVector(new THREE.Vector3(...part.explode_mm),value/1000);}}
function reassemble(){animated=false;disassembling=false;$('explode').value=0;setExplosion(0);$('animate').textContent='▷ 示意爆炸动画';$('disassemble').textContent='▷ 采样核验拆解';}
function resetCamera(){
  if(!model||!controls)return;reassemble();
  const bounds=new THREE.Box3().setFromObject(model),center=bounds.getCenter(new THREE.Vector3()),extent=Math.max(.001,bounds.getSize(new THREE.Vector3()).length());
  controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(1.05,1.45,1.1).multiplyScalar(extent*1.15));camera.near=Math.max(.0001,extent/100);camera.far=Math.max(5,extent*30);camera.updateProjectionMatrix();if(floorGrid)floorGrid.position.z=bounds.min.z-extent*.03;controls.update();
}
try{
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.setClearColor(0x000000,0);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.1;viewport.appendChild(renderer.domElement);
  controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=.02;controls.maxDistance=6;
  scene.add(new THREE.HemisphereLight(0xffffff,0x8eaa92,1.8));const key=new THREE.DirectionalLight(0xffffff,2.8);key.position.set(.3,.4,.5);scene.add(key);const fill=new THREE.DirectionalLight(0xcee1e5,2);fill.position.set(-.3,.2,.3);scene.add(fill);
  floorGrid=new THREE.GridHelper(.5,25,0xcad5c7,0xe1e7db);floorGrid.rotation.x=Math.PI/2;floorGrid.material.transparent=true;floorGrid.material.opacity=.5;scene.add(floorGrid);
  new ResizeObserver(()=>{const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}).observe(viewport);
  const raycaster=new THREE.Raycaster();renderer.domElement.addEventListener('click',event=>{const r=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1),camera);const hit=raycaster.intersectObjects([...meshes.values()],true)[0];if(hit){let object=hit.object;while(object&&!meshes.has(object.name))object=object.parent;if(object)selectPart(object.name,event.shiftKey);}});
  function frame(t){requestAnimationFrame(frame);if(animated){const v=(1-Math.cos((t-animationStart)/2200))/2;$('explode').value=Math.round(v*100);setExplosion(v);}if(disassembling&&structure?.disassembly){const moves=structure.disassembly.moves,progress=(t-animationStart)/1200;for(let i=0;i<moves.length;i++){const object=meshes.get(moves[i].part);if(object)object.position.copy(object.userData.originalPosition).addScaledVector(new THREE.Vector3(...moves[i].translation_mm),Math.min(1,Math.max(0,progress-i))/1000);}if(progress>=moves.length){disassembling=false;$('disassemble').textContent='↺ 重播采样核验拆解';}}controls.update();renderer.render(scene,camera);}requestAnimationFrame(frame);
}catch(error){$('empty').textContent='当前浏览器无法启用 3D 渲染，仍可生成与下载。';console.error(error);}
$('reset-camera').addEventListener('click',resetCamera);$('reassemble').addEventListener('click',reassemble);
$('explode').addEventListener('input',()=>{animated=false;disassembling=false;setExplosion(Number($('explode').value)/100);});
$('animate').addEventListener('click',()=>{animated=!animated;disassembling=false;animationStart=performance.now();$('animate').textContent=animated?'Ⅱ 暂停示意动画':'▷ 示意爆炸动画';});
$('disassemble').addEventListener('click',()=>{animated=false;disassembling=true;setExplosion(0);animationStart=performance.now();$('disassemble').textContent='采样核验序列播放中';});

function renderTree(){
  $('tree').replaceChildren();
  function children(parent,element){
    for(const group of structure.groups.filter(g=>g.parent===parent)){const div=document.createElement('div');div.className='group';div.textContent=group.label;element.appendChild(div);const list=document.createElement('ul');element.appendChild(list);children(group.id,list);}
    for(const part of structure.parts.filter(p=>p.parent===parent)){const row=document.createElement('div');row.className='part-row';const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.dataset.partCheck=part.id;checkbox.setAttribute('aria-label',`选择 ${part.label}`);checkbox.addEventListener('change',()=>selectPart(part.id,true));const button=document.createElement('button');button.dataset.part=part.id;button.textContent=`◦ ${part.label}`;button.addEventListener('click',e=>selectPart(part.id,e.shiftKey));row.append(checkbox,button);element.appendChild(row);}
  }children('root',$('tree'));updateSelection();
}
async function loadRun(id){
  const version=++loadingVersion,base=runBase(id),[summary,data,trace]=await Promise.all([json(base+'run.json'),json(base+'structure.json'),json(base+'trace.json')]);if(version!==loadingVersion)return;
  currentRun=id;currentSummary=summary;structure=data;selected.clear();reassemble();
  const imageDerived=summary.spec.family==='image_parts',meshOnly=data.representation==='generated_mesh_parts',planner=summary.settings.planner||summary.settings;setMode(imageDerived?'image':'param');
  if(imageDerived){$('image-parts').value=summary.settings.desired_parts;$('image-size').value=summary.settings.size_mm;$('image-resolution').value=summary.settings.resolution;$('remove-background').checked=summary.settings.remove_background;}
  else{chooseFamily(summary.spec.family);$('load').value=summary.spec.load;$('groups').value=summary.spec.groups;$('seed').value=summary.spec.seed;$('size').value=family==='drive'?summary.spec.shaft_length_mm:summary.spec.opening_mm;$('intent').value=summary.spec.source_text||'';}
  $('backend').value=[...$('backend').options].some(o=>o.value===planner.backend)?planner.backend:'auto';$('encoding').value=planner.encoding||'compact_binary';
  $('model-title').textContent=imageDerived?'从二维外观，探索三维结构。':family==='drive'?'轴、轴承与齿轮的结构协作。':'让夹持与导向各得其位。';$('representation-label').textContent=meshOnly?'3D / GENERATED MESH PARTS':'CAD / PARAMETRIC B-REP';
  $('part-count').textContent=summary.part_count;$('qubits').textContent=summary.resources.logical_qubits??'经典';$('shots').textContent=summary.resources.shots?.toLocaleString()??'—';$('duration').textContent=summary.timings.end_to_end_s.toFixed(2)+' s';
  $('validation-tag').textContent=summary.validation.valid?(meshOnly?'● 分件 / 合并校验通过':'● 几何校验通过'):'校验未通过';$('validation-tag').classList.toggle('valid',summary.validation.valid);$('selected-bits').textContent=summary.selected_bits;
  const source=summary.provenance.execution_source;$('source-tag').textContent=source==='classical_cpu'?'经典对照':source.includes('simulator')?'QB 模拟测量':'远端后端';
  const device=trace.flatMap(r=>r.plan?.trace||[]).findLast(b=>b.metadata?.device)?.metadata.device,label=device==='mps'?'Apple GPU / MPS':device==='cuda'?'CUDA GPU':device||'CPU';if(apiAvailable)$('runtime').textContent=`QB · ${source==='classical_cpu'?'经典 CPU':label}`;
  const compact=summary.resources.encoding==='compact_binary',physical=compact?summary.selected_bits.match(/../g).map(p=>p[1]).join(''):summary.selected_bits;
  $('trace-note').textContent=`${summary.provenance.selected_measurement?.kind==='conditioned_composite'?'分块条件测量组成结构 · ':compact?'测量 '+physical+' → 结构编码 · ':''}${summary.provenance.generation_rounds} 轮生成 · ${summary.provenance.fallback_used?'发生经典回退':'未发生经典回退'}${summary.kind==='mesh_edit'?' · 合并沿用原分件测量':''}`;
  $('disassemble').disabled=!data.disassembly?.moves?.length||data.parts.length<2;$('disassemble').textContent='▷ 采样核验拆解';$('restore-parts').disabled=!summary.original_run_id;
  $('source-reference').hidden=!imageDerived;if(imageDerived){$('source-image').src=base+'input.png';$('image-scope').textContent=`由 TripoSR 生成，标尺 ${summary.settings.size_mm} mm。背面、内部与分件关系为推测；真实尺寸和连接需确认。`;}
  $('decisions').replaceChildren();for(const[key,value]of Object.entries(summary.decisions)){const chip=document.createElement('div');chip.className='decision';const name=document.createElement('small');name.textContent=key;chip.append(name,document.createTextNode(value.label));$('decisions').appendChild(chip);}
  $('trace-panel').replaceChildren();for(const round of trace){const title=document.createElement('div');title.textContent=`ROUND ${round.round+1} · ${round.plan.execution_source}`;$('trace-panel').appendChild(title);const counts={};for(const batch of round.plan.trace)for(const[bits,n]of Object.entries(batch.counts||{}))counts[bits]=(counts[bits]||0)+n;for(const[bits,count]of Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,10)){const div=document.createElement('div');div.textContent=`${bits} × ${count}${bits===summary.selected_bits?' ← selected':''}`;$('trace-panel').appendChild(div);}}
  renderTree();for(const[element,file]of [['download-step',meshOnly?'assembly.obj':'assembly.step'],['download-glb','assembly.glb'],['download-json','structure.json'],['download-parts','parts.zip']]){$(element).href=base+file;$(element).setAttribute('aria-disabled','false');$(element).download=file;}$('download-step').textContent=meshOnly?'OBJ 网格 ↗':'STEP 实体 ↗';$('download-parts').hidden=!meshOnly;
  if(renderer){const gltf=await new GLTFLoader().loadAsync(base+'assembly.glb');if(version!==loadingVersion)return;if(model){scene.remove(model);model.traverse(n=>{if(n.isMesh){n.geometry.dispose();for(const m of Array.isArray(n.material)?n.material:[n.material])m.dispose();}});}meshes.clear();model=gltf.scene;scene.add(model);for(const part of data.parts){const object=model.getObjectByName(part.id);if(object){object.userData.originalPosition=object.position.clone();meshes.set(part.id,object);}}$('empty').hidden=true;resetCamera();}
  status(`${summary.part_count} 个${imageDerived?'候选分件':'独立零件'}已载入，可拆解、重新装配或选择合并。`);
}
$('toggle-trace').addEventListener('click',()=>{$('trace-panel').hidden=!$('trace-panel').hidden;});
document.querySelectorAll('[data-example]').forEach(b=>b.addEventListener('click',()=>loadRun(b.dataset.example).catch(e=>status(e.message,true))));
$('image-file').addEventListener('change',()=>{if(previewURL)URL.revokeObjectURL(previewURL);const file=$('image-file').files[0];$('image-preview').hidden=!file;if(file){previewURL=URL.createObjectURL(file);$('image-preview').src=previewURL;status(`已选择 ${file.name}，可生成三维外观与候选分件。`);}});
const dropZone=document.querySelector('.upload-label');
dropZone.addEventListener('dragover',event=>event.preventDefault());
dropZone.addEventListener('drop',event=>{event.preventDefault();if(event.dataTransfer.files.length){$('image-file').files=event.dataTransfer.files;$('image-file').dispatchEvent(new Event('change'));}});
async function waitJob(job){for(let i=0;i<1800;i++){await new Promise(resolve=>setTimeout(resolve,1000));const state=await json(`/api/jobs/${job.id}`);if(state.phase)status(state.phase+'…');if(state.status==='completed'){await loadRun(job.id);return;}if(state.status==='failed')throw new Error(state.error);}throw new Error(`等待较久，任务 ${job.id} 仍可在服务端继续运行。`);}
function requireAPI(){if(apiAvailable)return true;$('connection-panel').hidden=false;status('请先连接本机生成服务，再生成或保存合并结果。',true);return false;}
$('generate').addEventListener('click',async()=>{
  if(!requireAPI())return;busy=true;$('generate').disabled=true;
  try{let job;if(mode==='image'){
    const file=$('image-file').files[0];if(!file)throw new Error('请先选择一张二维图片');if(!modelReady)throw new Error('本地图像模型尚未准备完成，请先完成模型下载');const form=new FormData();form.append('file',file);form.append('options',JSON.stringify({resolution:Number($('image-resolution').value),desired_parts:Number($('image-parts').value),size_mm:Number($('image-size').value),remove_background:$('remove-background').checked,planner:{backend:$('backend').value,encoding:'compact_binary',iterations:6,depth:1,shots:512}}));status('正在上传图片…');job=await json('/api/generate-image',{method:'POST',body:form});
  }else{let spec={family,load:$('load').value,groups:Number($('groups').value),seed:Number($('seed').value),[family==='drive'?'shaft_length_mm':'opening_mm']:Number($('size').value)};if($('intent').value.trim()){const parsed=await json('/api/parse',{method:'POST',body:JSON.stringify({text:$('intent').value})});spec={...parsed.spec,groups:spec.groups,seed:spec.seed};}job=await json('/api/generate',{method:'POST',body:JSON.stringify({spec,settings:{backend:$('backend').value,encoding:$('encoding').value,iterations:12,shots:512}})});}status('任务已提交，正在生成…');await waitJob(job);
  }catch(e){status(e.message,true);}finally{busy=false;$('generate').disabled=false;updateSelection();}
});
$('merge-parts').addEventListener('click',async()=>{if(!requireAPI()||selected.size<2)return;busy=true;updateSelection();try{const job=await json(`/api/runs/${currentRun}/merge`,{method:'POST',body:JSON.stringify({part_ids:[...selected]})});await waitJob(job);}catch(e){status(e.message,true);}finally{busy=false;updateSelection();}});
$('restore-parts').addEventListener('click',()=>{if(currentSummary?.original_run_id)loadRun(currentSummary.original_run_id).catch(e=>status(e.message,true));});
async function refreshModel(){if(!apiAvailable)return;try{const data=await json('/api/image-model');modelReady=data.ready;$('image-model-status').textContent=data.ready?'TripoSR 已准备 · 本机生成 / QB 分件':`图像模型准备中 · ${(100*data.downloaded_bytes/data.checkpoint_bytes).toFixed(1)}% 权重已下载`;}catch{modelReady=false;}}
async function connect(){const url=new URL($('backend-url').value);if(!['http:','https:'].includes(url.protocol))throw new Error('请输入 http 或 https 服务地址');apiOrigin=url.href.replace(/\/$/,'');const info=await json('/api/status');apiAvailable=true;$('public-notice').hidden=true;$('connection-panel').hidden=true;try{localStorage.setItem('qamg-api',apiOrigin);}catch{}$('runtime').textContent=`QB 本地模拟 · ${info.runtime.selected_simulator_device}`;await refreshModel();return await json('/api/runs');}
$('connect-api').addEventListener('click',()=>{$('connection-panel').hidden=!$('connection-panel').hidden;});
$('save-connection').addEventListener('click',async()=>{try{const runs=await connect();status('生成服务已连接。');if(!currentRun&&runs.length)await loadRun(runs.find(r=>r.summary.kind==='image3d')?.id||runs[0].id);}catch(e){apiAvailable=false;status(`连接失败：${e.message}`,true);}});
async function staticGallery(){ $('public-notice').hidden=false;$('runtime').textContent='QB 生成示例 · 在线预览';$('image-model-status').textContent='生成新模型需要连接本机 GPU / CPU 服务';const data=await json(BASE+'gallery/index.json');for(const run of data.examples)staticRuns.set(run.id,run);if(data.examples.length)await loadRun(data.examples[0].id);}
try{if(localPage){try{const runs=await connect();if(runs.length)await loadRun(runs.find(r=>r.summary.kind==='image3d')?.id||runs[0].id);}catch{await staticGallery();}}else await staticGallery();}catch(e){status(e.message,true);}
setInterval(refreshModel,10000);
