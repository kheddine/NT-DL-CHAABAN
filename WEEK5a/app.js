// app.js
// Controller: load CSV → build tensors → train CNN+GRU → predict → visualize

import {DataLoader} from './data-loader.js';
import {CNN_GRU_Model} from './cnn_gru.js';

class StockApp{
  constructor(){
    this.loader=new DataLoader(); this.model=null; this.data=null; this.chart=null;
    this.initUI();
  }
  initUI(){
    csvFile.addEventListener('change',e=>this.load(e));
    trainBtn.addEventListener('click',()=>this.train());
    predictBtn.addEventListener('click',()=>this.predict());
  }
  status(m){document.getElementById('status').textContent=m;console.log(m);}

  async load(e){
    const f=e.target.files[0]; if(!f)return;
    try{
      this.status('Loading CSV…');
      await this.loader.loadCSV(f);
      this.status('Creating 12-day sequences…');
      this.data=this.loader.createSequences(12,3);
      trainBtn.disabled=false;
      this.status('Data ready. Click Train.');
    }catch(err){this.status('Error:'+err.message);console.error(err);}
  }

  async train(){
    const {X_train,y_train,X_val,y_val,symbols}=this.data;
    try{
      this.status('Training model…');
      this.model=new CNN_GRU_Model([12,2*symbols.length],3*symbols.length);
      await this.model.train(X_train,y_train,X_val,y_val,50,64);
      predictBtn.disabled=false;
      this.status('Training done. Run prediction.');
    }catch(err){this.status('Train error:'+err.message);console.error(err);}
  }

  async predict(){
    const {X_test,y_test,symbols}=this.data;
    try{
      this.status('Predicting…');
      const p=this.model.predict(X_test);
      const ev=this.model.evaluate(y_test,p,symbols);
      if(p.dispose)p.dispose();
      this.render(ev);
      this.status('Prediction complete.');
    }catch(err){this.status('Predict error:'+err.message);console.error(err);}
  }

  render(ev){
    const ctx=document.getElementById('accuracyChart').getContext('2d');
    if(this.chart)this.chart.destroy();
    const arr=Object.entries(ev.stockAccuracies).sort((a,b)=>b[1]-a[1]);
    this.chart=new Chart(ctx,{
      type:'bar',
      data:{labels:arr.map(a=>a[0]),
        datasets:[{label:'Accuracy (%)',data:arr.map(a=>a[1]*100),
          backgroundColor:arr.map(a=>a[1]>0.6?'#6be4c1cc':'#ff6b6bcc')}]},
      options:{indexAxis:'y',scales:{x:{beginAtZero:true,max:100}}}
    });
    const cont=document.getElementById('timelineContainer'); cont.innerHTML='';
    Object.entries(ev.stockPredictions).slice(0,3).forEach(([sym,preds])=>{
      const d=document.createElement('div'); d.className='stock-chart';
      d.innerHTML=`<h4>${sym}</h4><canvas id="tl-${sym}"></canvas>`;
      cont.appendChild(d);
      const c=document.getElementById(`tl-${sym}`).getContext('2d');
      new Chart(c,{
        type:'line',
        data:{labels:preds.slice(0,50).map((_,i)=>i+1),
          datasets:[{label:'Correct (1)/Wrong (0)',
            data:preds.slice(0,50).map(p=>p.correct?1:0),
            borderColor:'#6be4c1',backgroundColor:'#6be4c140',tension:0.3}]},
        options:{scales:{y:{min:0,max:1}}}
      });
    });
  }
}
document.addEventListener('DOMContentLoaded',()=>new StockApp());
