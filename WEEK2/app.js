/*  Titanic Binary Classifier – TensorFlow.js
 *  Author: You
 *  Runs fully in browser; ready for GitHub Pages.
 *  Reusable: swap schema section to adapt for new datasets.
 */

let trainData, testData, model;
let trainTensor, valTensor;
let threshold = 0.5;
const thSlider = document.getElementById('thSlider');
const thVal = document.getElementById('thVal');

// === Utility helpers ===
const csvToArray = async (file) => {
  const text = await file.text();
  // Simple split; commas inside quotes handled by regex
  return tf.data.csv(tf.util.toString(text), { hasHeader: true }).toArray();
};
const mean = arr => arr.reduce((a,b)=>a+b,0)/arr.length;

// === Data loading ===
document.getElementById('loadBtn').onclick = async () => {
  const trainFile = document.getElementById('trainFile').files[0];
  const testFile  = document.getElementById('testFile').files[0];
  if (!trainFile || !testFile) { alert('Select both CSVs'); return; }

  const t1 = await trainFile.text(), t2 = await testFile.text();
  trainData = Papa.parse(t1, {header:true, dynamicTyping:true}).data;
  testData  = Papa.parse(t2,  {header:true, dynamicTyping:true}).data;

  document.getElementById('dataInfo').textContent =
    `Loaded ${trainData.length} train rows, ${testData.length} test rows.`;

  // Preview head
  const head = trainData.slice(0,5);
  const keys = Object.keys(head[0]);
  const table = ['<table><tr>'+keys.map(k=>`<th>${k}</th>`).join('')+'</tr>'];
  head.forEach(r=>{
    table.push('<tr>'+keys.map(k=>`<td>${r[k]}</td>`).join('')+'</tr>');
  });
  document.getElementById('preview').innerHTML = table.join('')+'</table>';

  document.getElementById('prepBtn').disabled = false;

  // Quick tfjs-vis bars: survival by Sex / Pclass
  const bySex = {};
  trainData.forEach(r=>{
    if(r.Sex && (r.Survived===0||r.Survived===1)){
      bySex[r.Sex]=(bySex[r.Sex]||[]).concat(r.Survived);
    }
  });
  const sexStats = Object.keys(bySex).map(s=>({
    x:s, y:mean(bySex[s])
  }));
  tfvis.render.barchart({name:'Survival Rate by Sex', tab:'Data'}, sexStats);

  const byCls = {};
  trainData.forEach(r=>{
    if(r.Pclass && (r.Survived===0||r.Survived===1)){
      byCls[r.Pclass]=(byCls[r.Pclass]||[]).concat(r.Survived);
    }
  });
  const clsStats = Object.keys(byCls).map(s=>({
    x:`Class ${s}`, y:mean(byCls[s])
  }));
  tfvis.render.barchart({name:'Survival Rate by Pclass', tab:'Data'}, clsStats);
};

// === Preprocessing ===
document.getElementById('prepBtn').onclick = () => {
  // ---- SCHEMA (swap here for other datasets) ----
  const ageArr = trainData.map(r=>r.Age).filter(a=>!isNaN(a));
  const ageMed = tf.tensor1d(ageArr).median().arraySync();
  const embMode = ['S','C','Q'].sort(
    (a,b)=>trainData.filter(r=>r.Embarked===b).length
          -trainData.filter(r=>r.Embarked===a).length
  )[0];
  const fareMean = mean(trainData.map(r=>r.Fare||0));
  // ----------------------------------------------

  const proc = (row,isTrain=true)=>{
    let age = row.Age??ageMed;
    let fare = row.Fare??fareMean;
    const embarked = row.Embarked??embMode;
    const sex = row.Sex==='male'?1:0;
    // One-hot encoding: Pclass, Sex, Embarked
    const pclass=[0,0,0]; if(row.Pclass>=1&&row.Pclass<=3)pclass[row.Pclass-1]=1;
    const emb=[embarked==='S'?1:0, embarked==='C'?1:0, embarked==='Q'?1:0];
    const familySize=(row.SibSp??0)+(row.Parch??0)+1;
    const isAlone = familySize===1?1:0;

    return {
      x:[sex, age, fare, ...pclass, ...emb, familySize, isAlone],
      y:isTrain?row.Survived:undefined,
      id:row.PassengerId
    };
  };

  trainData=trainData.map(r=>proc(r,true)).filter(r=>r.y!==undefined);
  testData =testData.map(r=>proc(r,false));

  // Standardize Age/Fare
  const ages=trainData.map(r=>r.x[1]), fares=trainData.map(r=>r.x[2]);
  const meanA=mean(ages), sdA=Math.sqrt(mean(ages.map(a=>(a-meanA)**2)));
  const meanF=mean(fares), sdF=Math.sqrt(mean(fares.map(f=>(f-meanF)**2)));
  trainData.forEach(r=>{r.x[1]=(r.x[1]-meanA)/sdA; r.x[2]=(r.x[2]-meanF)/sdF;});
  testData.forEach(r=>{r.x[1]=(r.x[1]-meanA)/sdA; r.x[2]=(r.x[2]-meanF)/sdF;});

  document.getElementById('prepInfo').textContent =
    `Features = ${trainData[0].x.length}, Train samples = ${trainData.length}`;
  document.getElementById('buildBtn').disabled = false;
};

// === Model definition ===
document.getElementById('buildBtn').onclick = () => {
  model = tf.sequential();
  model.add(tf.layers.dense({inputShape:[trainData[0].x.length], units:16, activation:'relu'}));
  model.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  model.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});

  const lines=[];
  model.summary(100, undefined, s=>lines.push(s));
  document.getElementById('modelSummary').textContent=lines.join('\n');
  document.getElementById('trainBtn').disabled=false;
};

// === Training ===
document.getElementById('trainBtn').onclick = async ()=>{
  const xs=tf.tensor2d(trainData.map(r=>r.x));
  const ys=tf.tensor2d(trainData.map(r=>[r.y]));

  // 80/20 split
  const split=Math.floor(0.8*xs.shape[0]);
  const xTrain=xs.slice(0,split);
  const yTrain=ys.slice(0,split);
  const xVal=xs.slice(split);
  const yVal=ys.slice(split);

  const surface={name:'Training Progress',tab:'Training'};
  await model.fit(xTrain,yTrain,{
    epochs:50,batchSize:32,validationData:[xVal,yVal],
    callbacks:[
      tfvis.show.fitCallbacks(surface,['loss','val_loss','acc','val_acc']),
      tf.callbacks.earlyStopping({monitor:'val_loss',patience:5})
    ]
  });
  xs.dispose(); ys.dispose();
  document.getElementById('evalBtn').disabled=false;
};

// === Evaluation metrics ===
document.getElementById('evalBtn').onclick=async()=>{
  const xs=tf.tensor2d(trainData.map(r=>r.x));
  const ys=tf.tensor1d(trainData.map(r=>r.y));
  const probs=model.predict(xs).reshape([ys.shape[0]]);
  const yTrue=await ys.array(), yScore=await probs.array();
  xs.dispose(); ys.dispose(); probs.dispose();

  // Compute ROC curve
  const points=[];
  for(let t=0;t<=1;t+=0.02){
    let tp=0,fp=0,fn=0,tn=0;
    yScore.forEach((p,i)=>{
      const pred=p>=t?1:0;
      const y=yTrue[i];
      if(pred===1&&y===1)tp++;
      else if(pred===1&&y===0)fp++;
      else if(pred===0&&y===1)fn++;
      else tn++;
    });
    const tpr=tp/(tp+fn); const fpr=fp/(fp+tn);
    points.push({x:fpr,y:tpr});
  }
  tfvis.render.linechart({name:'ROC Curve',tab:'Metrics'}, {values:points}, {xLabel:'FPR',yLabel:'TPR'});
  thSlider.disabled=true;
  thSlider.disabled=false;
  document.getElementById('predBtn').disabled=false;
};

// === Threshold slider ===
thSlider.oninput=()=>{
  threshold=parseFloat(thSlider.value);
  thVal.textContent=threshold.toFixed(2);
  updateConfusion();
};

async function updateConfusion(){
  if(!model||!trainData)return;
  const xs=tf.tensor2d(trainData.map(r=>r.x));
  const ys=trainData.map(r=>r.y);
  const preds=model.predict(xs).reshape([ys.length]);
  const p=await preds.array();
  xs.dispose(); preds.dispose();
  let tp=0,fp=0,fn=0,tn=0;
  p.forEach((v,i)=>{
    const pr=v>=threshold?1:0; const y=ys[i];
    if(pr===1&&y===1)tp++; else if(pr===1&&y===0)fp++;
    else if(pr===0&&y===1)fn++; else tn++;
  });
  const prec=tp/(tp+fp+1e-9), rec=tp/(tp+fn+1e-9);
  const f1=2*prec*rec/(prec+rec+1e-9);
  document.getElementById('metricPlots').innerHTML=
  `<table>
    <tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
    <tr><th>Actual 0</th><td>${tn}</td><td>${fp}</td></tr>
    <tr><th>Actual 1</th><td>${fn}</td><td>${tp}</td></tr>
   </table>
   <p>Precision = ${prec.toFixed(3)}  Recall = ${rec.toFixed(3)}  F1 = ${f1.toFixed(3)}</p>`;
}

// === Prediction + Export ===
document.getElementById('predBtn').onclick=async()=>{
  const xs=tf.tensor2d(testData.map(r=>r.x));
  const preds=model.predict(xs).reshape([testData.length]);
  const p=await preds.array();
  xs.dispose(); preds.dispose();
  testData.forEach((r,i)=>r.prob=p[i]);
  document.getElementById('predInfo').textContent=`Predicted ${p.length} samples.`;
  document.getElementById('expBtn').disabled=false;
};

document.getElementById('expBtn').onclick=()=>{
  const lines=['PassengerId,Survived,Probability'];
  testData.forEach(r=>{
    const s=r.prob>=threshold?1:0;
    lines.push(`${r.id},${s},${r.prob.toFixed(5)}`);
  });
  const blob=new Blob([lines.join('\n')],{type:'text/csv'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='submission.csv';
  a.click();
  model.save('downloads://titanic-tfjs');
};
