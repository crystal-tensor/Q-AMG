import './style.css';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

const $=id=>document.getElementById(id), BASE=import.meta.env.BASE_URL;
const localPage=['127.0.0.1','localhost','[::1]'].includes(location.hostname);
const requestedRun=new URLSearchParams(location.search).get('run');
let apiOrigin=localPage?location.origin:'http://127.0.0.1:8765';
try{apiOrigin=localStorage.getItem('qamg-api')||apiOrigin;}catch{}
$('backend-url').value=apiOrigin;
let apiAvailable=false,modelReady=false,busy=false,mode='image',family='drive';
let structure,model,currentRun,currentSummary,animated=false,disassembling=false,animationStart=0,loadingVersion=0,previewURL;
let isolateSelected=false;
const meshes=new Map(),selected=new Set(),staticRuns=new Map();
const viewport=$('viewport'),scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(38,1,.001,100);
camera.up.set(0,0,1);
let renderer,controls,floorGrid;

function status(message,error=false){$('job-status').textContent=message;$('job-status').classList.toggle('error',error);}
function generationBusy(value){busy=value;$('generate').disabled=value;$('image-file').disabled=value;$('container-fit').disabled=value;document.querySelectorAll('[data-example],[data-mode]').forEach(button=>button.disabled=value);}
function clearRun(message='正在生成当前输入的模型…'){
  loadingVersion++;reassemble();
  if(model){scene.remove(model);model.traverse(n=>{if(n.isMesh){n.geometry.dispose();for(const material of Array.isArray(n.material)?n.material:[n.material])material.dispose();}});}
  model=undefined;structure=undefined;currentRun=undefined;currentSummary=undefined;meshes.clear();selected.clear();isolateSelected=false;
  $('empty').hidden=false;$('empty').textContent=message;$('tree').replaceChildren();
  for(const id of ['part-count','qubits','shots','duration','selected-bits'])$(id).textContent='—';
  $('validation-tag').textContent='尚无当前输入的结果';$('validation-tag').classList.remove('valid');
  $('source-reference').hidden=true;$('source-tag').textContent='等待本次结果';$('trace-note').textContent='本次生成完成后显示测量记录。';
  $('decisions').replaceChildren();$('trace-panel').replaceChildren();$('trace-panel').hidden=true;
  $('disassemble').disabled=true;$('restore-parts').disabled=true;
  $('container-fit-panel').hidden=true;
  for(const id of ['download-step','download-glb','download-json','download-parts']){$(id).removeAttribute('href');$(id).setAttribute('aria-disabled','true');}
  updateSelection();
}
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
  if(!selected.size)isolateSelected=false;
  for(const[id,object]of meshes)object.visible=!isolateSelected||selected.has(id);
  $('isolate-parts').disabled=!selected.size;$('isolate-parts').textContent=isolateSelected?'显示全部部件':'只看选中部件';
  for(const[id,object]of meshes)object.traverse(node=>{if(node.isMesh)for(const material of Array.isArray(node.material)?node.material:[node.material])if(material.emissive)material.emissive.setHex(selected.has(id)?0x184e37:0x000000);});
  document.querySelectorAll('[data-part]').forEach(b=>b.classList.toggle('active',selected.has(b.dataset.part)));
  document.querySelectorAll('[data-part-check]').forEach(b=>b.checked=selected.has(b.dataset.partCheck));
  $('merge-parts').disabled=selected.size<2||busy;$('selection-count').textContent=selected.size?`已选 ${selected.size} 个分件`:'在零件树勾选，或按 Shift 多选';
  const parts=structure?.parts.filter(p=>selected.has(p.id))||[];$('selection-label').hidden=!parts.length;
  $('part-label-editor').hidden=parts.length!==1;if(parts.length===1)$('part-label-input').value=parts[0].label;$('save-part-label').disabled=busy;
  if(parts.length)$('selection-label').textContent=`${parts.length===1?parts[0].label:parts.length+' 个分件'} · ${(parts.reduce((s,p)=>s+p.volume_mm3,0)/1000).toFixed(2)} cm³`;
}
function selectPart(id,toggle=false){if(!toggle)selected.clear();if(toggle&&selected.has(id))selected.delete(id);else selected.add(id);updateSelection();if(isolateSelected)resetCamera();}
function setExplosion(value){if(!structure)return;for(const part of structure.parts){const object=meshes.get(part.id);if(object)object.position.copy(object.userData.originalPosition).addScaledVector(new THREE.Vector3(...part.explode_mm),value/1000);}}
function reassemble(){animated=false;disassembling=false;$('explode').value=0;setExplosion(0);$('animate').textContent='▷ 示意爆炸动画';$('disassemble').textContent='▷ 采样核验拆解';}
function fitCamera(includeExploded=false){
  if(!model||!controls)return;
  const bounds=new THREE.Box3();if(isolateSelected){for(const id of selected){const object=meshes.get(id);if(object)bounds.union(new THREE.Box3().setFromObject(object));}}else bounds.setFromObject(model);const extent=Math.max(.001,bounds.getSize(new THREE.Vector3()).length());
  if(includeExploded&&structure){for(const part of structure.parts){const object=meshes.get(part.id);if(object&&(!isolateSelected||selected.has(part.id)))bounds.union(new THREE.Box3().setFromObject(object).translate(new THREE.Vector3(...part.explode_mm).multiplyScalar(.001)));}}
  const target=bounds.getCenter(new THREE.Vector3()),span=Math.max(extent,bounds.getSize(new THREE.Vector3()).length());
  controls.target.copy(target);camera.position.copy(target).add(new THREE.Vector3(1.05,1.45,1.1).multiplyScalar(span*1.15));camera.near=Math.max(.0001,span/100);camera.far=Math.max(5,span*30);camera.updateProjectionMatrix();if(floorGrid)floorGrid.position.z=bounds.min.z-span*.03;controls.update();
}
function resetCamera(){reassemble();fitCamera();}
try{
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.setClearColor(0x000000,0);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.1;viewport.appendChild(renderer.domElement);
  controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=.02;controls.maxDistance=6;
  scene.add(new THREE.HemisphereLight(0xffffff,0x8eaa92,1.8));const key=new THREE.DirectionalLight(0xffffff,2.8);key.position.set(.3,.4,.5);scene.add(key);const fill=new THREE.DirectionalLight(0xcee1e5,2);fill.position.set(-.3,.2,.3);scene.add(fill);
  floorGrid=new THREE.GridHelper(.5,25,0xcad5c7,0xe1e7db);floorGrid.rotation.x=Math.PI/2;floorGrid.material.transparent=true;floorGrid.material.opacity=.5;scene.add(floorGrid);
  new ResizeObserver(()=>{const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}).observe(viewport);
  const raycaster=new THREE.Raycaster();renderer.domElement.addEventListener('click',event=>{const r=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1),camera);const hit=raycaster.intersectObjects([...meshes.values()],true)[0];if(hit){let object=hit.object;while(object&&!meshes.has(object.name))object=object.parent;if(object)selectPart(object.name,event.shiftKey);}});
  function frame(t){requestAnimationFrame(frame);if(animated){const v=(1-Math.cos((t-animationStart)/2200))/2;$('explode').value=Math.round(v*100);setExplosion(v);}if(disassembling&&structure?.disassembly){const moves=structure.disassembly.moves,progress=(t-animationStart)/1200;for(let i=0;i<moves.length;i++){const object=meshes.get(moves[i].part);if(object)object.position.copy(object.userData.originalPosition).addScaledVector(new THREE.Vector3(...moves[i].translation_mm),Math.min(1,Math.max(0,progress-i))/1000);}if(progress>=moves.length){disassembling=false;$('disassemble').textContent='↺ 重播采样核验拆解';}}controls.update();renderer.render(scene,camera);}requestAnimationFrame(frame);
}catch(error){$('empty').textContent='当前浏览器无法启用 3D 渲染，仍可生成与下载。';console.error(error);}
$('reset-camera').addEventListener('click',resetCamera);$('reassemble').addEventListener('click',()=>{reassemble();fitCamera();});
$('isolate-parts').addEventListener('click',()=>{isolateSelected=!isolateSelected;updateSelection();resetCamera();});
$('explode').addEventListener('input',()=>{animated=false;disassembling=false;setExplosion(Number($('explode').value)/100);fitCamera();});
$('animate').addEventListener('click',()=>{animated=!animated;disassembling=false;if(animated){setExplosion(0);fitCamera(true);}animationStart=performance.now();$('animate').textContent=animated?'Ⅱ 暂停示意动画':'▷ 示意爆炸动画';});
$('disassemble').addEventListener('click',()=>{animated=false;disassembling=true;setExplosion(0);fitCamera(true);animationStart=performance.now();$('disassemble').textContent='采样核验序列播放中';});

function renderTree(){
  $('tree').replaceChildren();
  function children(parent,element){
    for(const group of structure.groups.filter(g=>g.parent===parent)){const div=document.createElement('div');div.className='group';div.textContent=group.label;element.appendChild(div);const list=document.createElement('ul');element.appendChild(list);children(group.id,list);}
    for(const part of structure.parts.filter(p=>p.parent===parent)){const row=document.createElement('div');row.className='part-row';const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.dataset.partCheck=part.id;checkbox.setAttribute('aria-label',`选择 ${part.label}`);checkbox.addEventListener('change',()=>selectPart(part.id,true));const button=document.createElement('button');button.dataset.part=part.id;button.textContent=`◦ ${part.label}`;button.addEventListener('click',e=>selectPart(part.id,e.shiftKey));row.append(checkbox,button);element.appendChild(row);}
  }children('root',$('tree'));updateSelection();
}
async function loadRun(id){
  clearRun('正在载入所选模型…');
  const version=++loadingVersion,base=runBase(id),[summary,data,trace]=await Promise.all([json(base+'run.json'),json(base+'structure.json'),json(base+'trace.json')]);if(version!==loadingVersion)return;
  if(summary.quality_review?.accepted===false)throw new Error(`此结果未通过部件核对：${summary.quality_review.reason}`);
  currentRun=id;currentSummary=summary;structure=data;selected.clear();reassemble();
  const imageDerived=summary.spec.family==='image_parts',meshOnly=data.representation==='generated_mesh_parts',planner=summary.settings.planner||summary.settings;setMode(imageDerived?'image':'param');
  if(imageDerived){$('image-parts').value=summary.settings.desired_parts;$('image-size').value=summary.settings.size_mm;$('image-resolution').value=summary.settings.resolution;$('image-seed').value=summary.settings.seed;$('remove-background').checked=summary.settings.remove_background;}
  else{chooseFamily(summary.spec.family);$('load').value=summary.spec.load;$('groups').value=summary.spec.groups;$('seed').value=summary.spec.seed;$('size').value=family==='drive'?summary.spec.shaft_length_mm:summary.spec.opening_mm;$('intent').value=summary.spec.source_text||'';}
  $('backend').value=[...$('backend').options].some(o=>o.value===planner.backend)?planner.backend:'auto';$('encoding').value=planner.encoding||'compact_binary';
  $('model-title').textContent=imageDerived?`${id.startsWith('example-')?'示例 · ':''}${summary.source_name||'图像生成结果'}`:family==='drive'?'轴、轴承与齿轮的结构协作。':'让夹持与导向各得其位。';$('representation-label').textContent=meshOnly?'3D / GENERATED MESH PARTS':'CAD / PARAMETRIC B-REP';
  $('part-count').textContent=summary.part_count;$('qubits').textContent=summary.resources.peak_logical_qubits??summary.resources.logical_qubits??'经典';$('shots').textContent=(summary.resources.total_shots??summary.resources.shots)?.toLocaleString()??'—';$('duration').textContent=summary.timings.end_to_end_s.toFixed(2)+' s';
  $('validation-tag').textContent=summary.validation.valid?(meshOnly?'● 网格有效 · 结构待核验':'● 几何校验通过'):'校验未通过';$('validation-tag').classList.toggle('valid',summary.validation.valid);$('selected-bits').textContent=summary.selected_bits;
  const source=summary.provenance.execution_source;$('source-tag').textContent=source==='classical_cpu'?'经典对照':source.includes('simulator')?'QB 模拟测量':'远端后端';
  const device=trace.flatMap(r=>r.plan?.trace||[]).findLast(b=>b.metadata?.device)?.metadata.device,label=device==='mps'?'Apple GPU / MPS':device==='cuda'?'CUDA GPU':device||'CPU';if(apiAvailable)$('runtime').textContent=`QB · ${source==='classical_cpu'?'经典 CPU':label}`;
  const compact=summary.resources.encoding==='compact_binary',physical=compact?summary.selected_bits.match(/../g).map(p=>p[1]).join(''):summary.selected_bits;
  $('trace-note').textContent=`${summary.provenance.selected_measurement?.kind==='conditioned_composite'?'分块条件测量组成结构 · ':compact?'测量 '+physical+' → 结构编码 · ':''}${summary.provenance.generation_rounds} 轮生成 · ${summary.provenance.fallback_used?'发生经典回退':'未发生经典回退'}${summary.kind==='mesh_edit'?' · 合并沿用原分件测量':''}`;
  $('disassemble').disabled=!data.disassembly?.moves?.length||data.parts.length<2;$('disassemble').textContent='▷ 采样核验拆解';$('restore-parts').disabled=!summary.original_run_id;
  $('container-fit-panel').hidden=!imageDerived;
  $('source-reference').hidden=!imageDerived;if(imageDerived){$('source-image').src=base+'input.png';const scope=summary.provenance.structure_prior==='explicit_lid_cup_sleeve'?'按盖子、杯子、杯套的显式结构约束辅助拟合；盖子提片为形状近似，连续空心杯体、壁厚、间隙及透明显示均含假设；':summary.provenance.geometric_slicing_performed===false?'独立部件由神经模型联合生成，按部件配色；':'历史几何切片结果，不代表真实部件识别；';const overlap=summary.validation.generated_part_overlap_fraction||0,moves=data.disassembly?.moves?.length||0;const concerns=`${overlap>1e-5?'部件交叠约 '+(100*overlap).toFixed(1)+'%；':''}${data.disassembly?.complete_for_generated_mesh===false?'仅 '+moves+'/'+Math.max(0,data.parts.length-1)+' 步通过当前方向的离散拆解检查；':''}`;$('image-scope').textContent=`${summary.model.model} · 标尺 ${summary.settings.size_mm} mm。${scope}${concerns}背面、内部与真实连接需核验。`;}
  $('decisions').replaceChildren();for(const[key,value]of Object.entries(summary.decisions)){const chip=document.createElement('div');chip.className='decision';const name=document.createElement('small');name.textContent=key;chip.append(name,document.createTextNode(value.label));$('decisions').appendChild(chip);}
  $('trace-panel').replaceChildren();for(const round of trace){const title=document.createElement('div');title.textContent=`ROUND ${round.round+1} · ${round.plan.execution_source}`;$('trace-panel').appendChild(title);const counts={};for(const batch of round.plan.trace)for(const[bits,n]of Object.entries(batch.counts||{}))counts[bits]=(counts[bits]||0)+n;for(const[bits,count]of Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,10)){const div=document.createElement('div');div.textContent=`${bits} × ${count}${bits===summary.selected_bits?' ← selected':''}`;$('trace-panel').appendChild(div);}}
  renderTree();for(const[element,file]of [['download-step',meshOnly?'assembly.obj':'assembly.step'],['download-glb','assembly.glb'],['download-json','structure.json'],['download-parts','parts.zip']]){$(element).href=base+file;$(element).setAttribute('aria-disabled','false');$(element).download=file;}$('download-step').textContent=meshOnly?'OBJ 网格 ↗':'STEP 实体 ↗';$('download-parts').hidden=!meshOnly;
  if(renderer){const gltf=await new GLTFLoader().loadAsync(base+'assembly.glb');if(version!==loadingVersion)return;if(model){scene.remove(model);model.traverse(n=>{if(n.isMesh){n.geometry.dispose();for(const m of Array.isArray(n.material)?n.material:[n.material])m.dispose();}});}meshes.clear();model=gltf.scene;scene.add(model);for(const part of data.parts){const object=model.getObjectByName(part.id);if(object){object.userData.originalPosition=object.position.clone();meshes.set(part.id,object);}}$('empty').hidden=true;resetCamera();}
  status(`${summary.part_count} 个${imageDerived?'候选分件':'独立零件'}已载入，可拆解、重新装配或选择合并。`);
}
$('toggle-trace').addEventListener('click',()=>{$('trace-panel').hidden=!$('trace-panel').hidden;});
document.querySelectorAll('[data-example]').forEach(b=>b.addEventListener('click',()=>loadRun(b.dataset.example).catch(e=>status(e.message,true))));
$('image-file').addEventListener('change',()=>{if(previewURL)URL.revokeObjectURL(previewURL);const file=$('image-file').files[0];$('image-preview').hidden=!file;if(file){clearRun('已选择新图片，请生成后查看对应三维结果。');previewURL=URL.createObjectURL(file);$('image-preview').src=previewURL;status(`已选择 ${file.name}，可生成三维外观与候选分件。`);}});
const dropZone=document.querySelector('.upload-label');
dropZone.addEventListener('dragover',event=>event.preventDefault());
dropZone.addEventListener('drop',event=>{event.preventDefault();if(event.dataTransfer.files.length){$('image-file').files=event.dataTransfer.files;$('image-file').dispatchEvent(new Event('change'));}});
async function waitJob(job){for(let i=0;i<1800;i++){await new Promise(resolve=>setTimeout(resolve,1000));const state=await json(`/api/jobs/${job.id}`);if(state.phase)status(state.phase+'…');if(state.status==='completed'){await loadRun(job.id);return;}if(state.status==='failed')throw new Error(state.error);}throw new Error(`等待较久，任务 ${job.id} 仍可在服务端继续运行。`);}
function requireAPI(){if(apiAvailable)return true;$('connection-panel').hidden=false;status('请先连接本机生成服务，再生成或保存合并结果。',true);return false;}
$('generate').addEventListener('click',async()=>{
  if(!requireAPI())return;generationBusy(true);
  try{let job;if(mode==='image'){
    const file=$('image-file').files[0];if(!file)throw new Error('请先选择一张二维图片');if(!modelReady)throw new Error('本地图像模型尚未准备完成，请先完成模型下载');clearRun(`正在生成 ${file.name} 的三维结果…`);const form=new FormData();form.append('file',file);form.append('options',JSON.stringify({resolution:Number($('image-resolution').value),desired_parts:Number($('image-parts').value),seed:Number($('image-seed').value),size_mm:Number($('image-size').value),remove_background:$('remove-background').checked,planner:{backend:$('backend').value,encoding:'compact_binary',iterations:6,depth:1,shots:512}}));status('正在上传图片…');job=await json('/api/generate-image',{method:'POST',body:form});
  }else{let spec={family,load:$('load').value,groups:Number($('groups').value),seed:Number($('seed').value),[family==='drive'?'shaft_length_mm':'opening_mm']:Number($('size').value)};if($('intent').value.trim()){const parsed=await json('/api/parse',{method:'POST',body:JSON.stringify({text:$('intent').value})});spec={...parsed.spec,groups:spec.groups,seed:spec.seed};}job=await json('/api/generate',{method:'POST',body:JSON.stringify({spec,settings:{backend:$('backend').value,encoding:$('encoding').value,iterations:12,shots:512}})});}status('任务已提交，正在生成…');await waitJob(job);
  }catch(e){if(!currentRun){$('empty').hidden=false;$('empty').textContent='本次生成失败，未显示其他图片的模型。';$('validation-tag').textContent='生成失败';}status(e.message,true);}finally{generationBusy(false);updateSelection();}
});
$('merge-parts').addEventListener('click',async()=>{if(!requireAPI()||selected.size<2)return;busy=true;updateSelection();try{const job=await json(`/api/runs/${currentRun}/merge`,{method:'POST',body:JSON.stringify({part_ids:[...selected]})});await waitJob(job);}catch(e){status(e.message,true);}finally{busy=false;updateSelection();}});
$('restore-parts').addEventListener('click',()=>{if(currentSummary?.original_run_id)loadRun(currentSummary.original_run_id).catch(e=>status(e.message,true));});
$('save-part-label').addEventListener('click',async()=>{if(!requireAPI()||selected.size!==1)return;const id=[...selected][0],label=$('part-label-input').value.trim();if(!label){status('请输入部件名称。',true);return;}generationBusy(true);updateSelection();try{const job=await json(`/api/runs/${currentRun}/labels`,{method:'POST',body:JSON.stringify({labels:{[id]:label}})});await waitJob(job);}catch(e){status(e.message,true);}finally{generationBusy(false);updateSelection();}});
$('container-fit').addEventListener('click',async()=>{if(!requireAPI()||!currentRun)return;generationBusy(true);try{status('按盖子、杯子、杯套约束拟合，保留原始参考…');const job=await json(`/api/runs/${currentRun}/container-fit`,{method:'POST'});await waitJob(job);}catch(e){status(e.message,true);}finally{generationBusy(false);updateSelection();}});
async function refreshModel(){if(!apiAvailable)return;try{const data=await json('/api/image-model');modelReady=data.ready;$('image-model-status').textContent=data.ready?`${data.model} 已准备 · 联合生成独立部件`:`部件模型准备中 · ${(100*data.downloaded_bytes/data.checkpoint_bytes).toFixed(1)}% 权重已下载`;}catch{modelReady=false;}}
async function connect(){const url=new URL($('backend-url').value);if(!['http:','https:'].includes(url.protocol))throw new Error('请输入 http 或 https 服务地址');apiOrigin=url.href.replace(/\/$/,'');const info=await json('/api/status');apiAvailable=true;$('public-notice').hidden=true;$('connection-panel').hidden=true;try{localStorage.setItem('qamg-api',apiOrigin);}catch{}$('runtime').textContent=`QB 本地模拟 · ${info.runtime.selected_simulator_device}`;await refreshModel();const runs=await json('/api/runs');$('robot-example').hidden=!runs.some(r=>r.id==='example-robot');return runs;}
$('connect-api').addEventListener('click',()=>{$('connection-panel').hidden=!$('connection-panel').hidden;});
$('save-connection').addEventListener('click',async()=>{try{await connect();status('生成服务已连接，请选择图片或示例。');}catch(e){apiAvailable=false;status(`连接失败：${e.message}`,true);}});
async function staticGallery(){ $('public-notice').hidden=false;$('runtime').textContent='QB 生成示例 · 在线预览';$('image-model-status').textContent='生成新模型需要连接本机 GPU / CPU 服务';const data=await json(BASE+'gallery/index.json');for(const run of data.examples)staticRuns.set(run.id,run);$('robot-example').hidden=!staticRuns.has('example-robot');if(data.examples.length)await loadRun(staticRuns.has(requestedRun)?requestedRun:data.examples[0].id);}
try{if(localPage){try{await connect();if(requestedRun)await loadRun(requestedRun);}catch(e){if(requestedRun)throw e;await staticGallery();}}else await staticGallery();}catch(e){status(e.message,true);}
setInterval(refreshModel,10000);
