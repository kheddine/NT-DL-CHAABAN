// data-loader.js
// Parse CSV → build 12-day Open/Close sequences → normalize → tensors

export class DataLoader {
  constructor() {
    this.symbols = [];
    this.dates = [];
    this.data = {};
  }

  async loadCSV(file) {
    const text = await file.text();
    const lines = text.split(/\r?\n/).filter(l=>l.trim().length);
    const headers = lines[0].split(',').map(h=>h.trim());
    const req = ['Date','Symbol','Open','Close'];
    for (const h of req) if (!headers.includes(h)) throw new Error(`Missing ${h}`);
    const idx = Object.fromEntries(headers.map((h,i)=>[h,i]));
    const data={},syms=new Set(),dts=new Set();
    for(let i=1;i<lines.length;i++){
      const v=lines[i].split(',');
      if(v.length<headers.length) continue;
      const s=v[idx.Symbol]?.trim(),d=v[idx.Date]?.trim();
      if(!s||!d) continue;
      const o=parseFloat(v[idx.Open]),c=parseFloat(v[idx.Close]);
      if(isNaN(o)||isNaN(c)) continue;
      if(!data[s]) data[s]={};
      data[s][d]={Open:o,Close:c};
      syms.add(s); dts.add(d);
    }
    this.symbols=Array.from(syms).sort().slice(0,10);
    this.dates=Array.from(dts).sort();
    this.data=data;
    if(this.symbols.length===0) throw new Error("No symbols found");
  }

  createSequences(seqLen=12,horizon=3){
    const Xraw=[],Y=[],syms=this.symbols;
    for(let i=seqLen;i<this.dates.length-horizon;i++){
      let valid=true; const window=[];
      for(let j=i-seqLen;j<i;j++){
        const d=this.dates[j]; const step=[];
        for(const s of syms){
          const r=this.data[s][d];
          if(!r){valid=false;break;}
          step.push(r.Open,r.Close);
        }
        if(!valid)break; window.push(step);
      }
      if(!valid)continue;
      const target=[];
      for(let h=1;h<=horizon;h++){
        const fd=this.dates[i+h];
        for(const s of syms){
          const cur=this.data[s][this.dates[i]], fut=this.data[s][fd];
          if(!cur||!fut){valid=false;break;}
          target.push(fut.Close>cur.Close?1:0);
        }
      }
      if(valid){Xraw.push(window);Y.push(target);}
    }
    if(!Xraw.length) throw new Error("No valid sequences");
    const n=Xraw.length;
    let nTrain=Math.floor(n*0.7),nVal=Math.floor(n*0.85);
    if(nVal===nTrain){nVal=nTrain+1;}
    const Xtr=Xraw.slice(0,nTrain),Xv=Xraw.slice(nTrain,nVal),Xte=Xraw.slice(nVal);
    const minMax={};
    syms.forEach((s,si)=>{
      let mn=Infinity,mx=-Infinity;
      for(const w of Xtr){for(const st of w){
        const base=si*2;
        for(let k=0;k<2;k++){const val=st[base+k];if(val<mn)mn=val;if(val>mx)mx=val;}
      }}
      if(!isFinite(mn)||!isFinite(mx)||mn===mx){mn=0;mx=1;}
      minMax[s]={min:mn,max:mx};
    });
    const scale=arr=>arr.map(w=>w.map(st=>{
      const out=[]; syms.forEach((s,si)=>{
        const {min,max}=minMax[s]; const base=si*2; const denom=Math.max(1e-9,max-min);
        for(let k=0;k<2;k++) out.push((st[base+k]-min)/denom);
      }); return out;
    }));
    const X_train=tf.tensor3d(scale(Xtr));
    const y_train=tf.tensor2d(Y.slice(0,nTrain));
    const X_val=tf.tensor3d(scale(Xv));
    const y_val=tf.tensor2d(Y.slice(nTrain,nVal));
    const X_test=tf.tensor3d(scale(Xte));
    const y_test=tf.tensor2d(Y.slice(nVal));
    return {X_train,y_train,X_val,y_val,X_test,y_test,symbols:syms};
  }
}
