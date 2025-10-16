// cnn_gru.js
// CNN + GRU hybrid model for binary stock prediction

export class CNN_GRU_Model {
  constructor(inputShape, outputSize){
    this.inputShape=inputShape;
    this.outputSize=outputSize;
    this.model=null;
  }

  async ensureBackend(){
    try{await tf.setBackend('webgl');await tf.ready();}
    catch(e){await tf.setBackend('cpu');await tf.ready();}
  }

  buildModel(){
    this.model=tf.sequential({
      layers:[
        tf.layers.conv1d({filters:64,kernelSize:3,activation:'relu',inputShape:this.inputShape}),
        tf.layers.dropout({rate:0.2}),
        tf.layers.gru({units:64}),
        tf.layers.dropout({rate:0.2}),
        tf.layers.dense({units:64,activation:'relu'}),
        tf.layers.dense({units:this.outputSize,activation:'sigmoid'})
      ]
    });
    this.model.compile({
      optimizer:tf.train.adam(0.001),
      loss:'binaryCrossentropy',
      metrics:['binaryAccuracy']
    });
    return this.model;
  }

  async train(Xtr,ytr,Xv,yv,epochs=50,batchSize=64){
    await this.ensureBackend();
    if(!this.model) this.buildModel();
    const valOk=(Xv?.shape?.[0]||0)>0 && (yv?.shape?.[0]||0)>0;
    const early=tf.callbacks.earlyStopping({monitor:valOk?'val_loss':'loss',patience:6,restoreBestWeights:true});
    const cb={
      onEpochEnd:async(e,l)=>{
        const p=document.getElementById('trainingProgress');const s=document.getElementById('status');
        if(p)p.value=((e+1)/epochs)*100;
        if(s)s.textContent=`Epoch ${e+1}/${epochs} loss:${l.loss.toFixed(4)} acc:${(l.binaryAccuracy*100).toFixed(1)}%`;
        await tf.nextFrame();
      }
    };
    return await this.model.fit(Xtr,ytr,{epochs,batchSize,shuffle:true,validationData:valOk?[Xv,yv]:null,callbacks:[early,cb]});
  }

  predict(X){const o=this.model.predict(X);return Array.isArray(o)?o[0]:o;}

  evaluate(yTrue,yPred,symbols,horizon=3){
    const t=yTrue.arraySync(),p=yPred.arraySync(),acc={},det={};
    symbols.forEach((sym,si)=>{
      let cor=0,tot=0;const arr=[];
      for(let i=0;i<t.length;i++){
        for(let h=0;h<horizon;h++){
          const idx=si*horizon+h;const tr=t[i][idx],pr=p[i][idx]>0.5?1:0;
          arr.push({truth:tr,pred:pr,correct:tr===pr});if(tr===pr)cor++;tot++;
        }}
      acc[sym]=tot?cor/tot:0;det[sym]=arr;
    });
    return {stockAccuracies:acc,stockPredictions:det};
  }
}
