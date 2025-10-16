// rnn_gru.js
// Hybrid SimpleRNN + GRU model with early stopping

export class RNN_GRU_Model {
  constructor(inputShape, outputSize){
    this.inputShape=inputShape;
    this.outputSize=outputSize;
    this.model=null;
  }

  buildModel(){
    this.model=tf.sequential({
      layers:[
        tf.layers.simpleRNN({units:64,returnSequences:true,inputShape:this.inputShape}),
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
    if(!this.model)this.buildModel();
    await tf.setBackend('webgl'); await tf.ready();

    const early=tf.callbacks.earlyStopping({
      monitor:'val_loss',patience:6,restoreBestWeights:true
    });

    return await this.model.fit(Xtr,ytr,{
      epochs,batchSize,validationData:[Xv,yv],shuffle:true,
      callbacks:[
        early,
        {onEpochEnd:async(ep,logs)=>{
          const p=document.getElementById('trainingProgress');
          const s=document.getElementById('status');
          if(p)p.value=((ep+1)/epochs)*100;
          if(s)s.textContent=
            `Epoch ${ep+1}/${epochs} | loss:${logs.loss.toFixed(4)} | acc:${(logs.binaryAccuracy*100).toFixed(1)}% | val_acc:${(logs.val_binaryAccuracy*100).toFixed(1)}%`;
          await tf.nextFrame();
        }}
      ]
    });
  }

  predict(X){return this.model.predict(X);}

  evaluate(yTrue,yPred,symbols,horizon=3){
    const t=yTrue.arraySync(),p=yPred.arraySync();
    const acc={},detail={};
    symbols.forEach((sym,si)=>{
      let corr=0,tot=0; const preds=[];
      for(let s=0;s<t.length;s++){
        for(let h=0;h<horizon;h++){
          const idx=si*horizon+h;
          const tr=t[s][idx],pr=p[s][idx]>0.5?1:0;
          preds.push({truth:tr,pred:pr,correct:tr===pr});
          if(tr===pr)corr++; tot++;
        }
      }
      acc[sym]=tot?corr/tot:0; detail[sym]=preds;
    });
    return {stockAccuracies:acc,stockPredictions:detail};
  }
}
