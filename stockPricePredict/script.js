function run(){
  let div = document.getElementById('main');

  const model = tf.sequential({
    name: "Stock Prediction Model",
    layers: [
      tf.layers.dense({units: 21, inputShape: [3], activation: 'sigmoid'}),
      tf.layers.dense({units: 21, activation: 'sigmoid'}),
      tf.layers.dense({units: 21, activation: 'sigmoid'}),
      tf.layers.dense({units: 3}),
    ]
  });

  const learningRate = 0.01;

  model.compile({optimizer: tf.train.sgd(learningRate), loss: 'meanSquaredError'});

  //stock per day(21 days), December (open, high, low)
  const googStockInput1 = tf.tensor2d([
    [1076.08, 1094.24, 1076.00],
    [1089.07, 1095.57, 1077.88],
    [1123.14, 1124.65, 1103.67],
    [1103.12, 1104.42, 1049.98],
    [1034.26, 1071.20, 1030.77],
    [1060.01, 1075.26, 1028.50],
    [1035.05, 1048.45, 1023.29],
    [1056.49, 1060.60, 1039.84],
    [1068.00, 1081.65, 1062.79],
    [1068.07, 1079.76, 1053.93],
    [1049.98, 1062.60, 1040.79],
    [1037.51, 1053.15, 1007.90],
    [1026.09, 1049.48, 1021.44],
    [1033.99, 1062.00, 1008.05],
    [1018.13, 1034.22, 996.36],
    [1015.30, 1024.02, 973.69],
    [973.90, 1003.54, 970.11],
    [989.01, 1040.00, 983.00],
    [1017.15, 1043.89, 997.00],
    [1049.62, 1055.56, 1033.10],
    [1050.96, 1052.70, 1023.59]
  ]).div(tf.tensor1d([2000, 2000, 2000]));

  //stock per day(21 days), Feb (open, high, low)
  const googStockOutput = tf.tensor2d([
    [1112.40, 1125.00, 1104.89],
    [1112.66, 1132.80, 1109.02],
    [1124.84, 1146.85, 1117.25],
    [1139.57, 1147.00, 1112.77],
    [1104.16, 1104.84, 1086.00],
    [1087.00, 1098.91, 1086.55],
    [1096.95, 1105.94, 1092.86],
    [1106.80, 1125.30, 1105.85],
    [1124.99, 1134.73, 1118.50],
    [1118.05, 1128.23, 1110.44],
    [1130.08, 1131.67, 1110.65],
    [1110.00, 1121.89, 1110.00],
    [1119.99, 1123.41, 1105.28],
    [1110.84, 1111.94, 1092.52],
    [1100.90, 1111.24, 1095.60],
    [1116.00, 1118.54, 1107.27],
    [1105.75, 1119.51, 1099.92],
    [1106.95, 1117.98, 1101.00],
    [1111.30, 1127.65, 1111.01],
    [1124.90, 1142.97, 1124.75],
    [1124.90, 1142.97, 1124.75]
  ]).div(tf.tensor1d([2000, 2000, 2000]));

  trainModel()
  .then(prediction)
  .then(predictionOnOutput)

  function prediction(){
    let guess = model.predict(googStockInput1);
    guess.mul(tf.tensor1d([2000, 2000, 2000])).print();
    let guessPred = guess.mul(tf.tensor1d([2000, 2000, 2000])).toString().replace(/\n/g, '</br>').replace(/\s/g, '&nbsp;&nbsp;').replace(/Tensor/g, '<b>Google Predicted Output This Month</b>').replace(/\[/g, '<font color="white"><i>[').replace(/]/g, ']</i></font>');
    div.innerHTML += guessPred + '<br>';

    return guess;
  }

  function predictionOnOutput(feedback){
    guess = model.predict(feedback);
    guess.mul(tf.tensor1d([2000, 2000, 2000])).print();
    let guessPred = guess.mul(tf.tensor1d([2000, 2000, 2000])).toString().replace(/\n/g, '</br>').replace(/\s/g, '&nbsp;&nbsp;').replace(/Tensor/g, '<b>Google Predicted Output Next Month</b>').replace(/\[/g, '<font color="white"><i>[').replace(/]/g, ']</i></font>');
    div.innerHTML += guessPred + '<br>';
    
    return guess;
  }

  async function trainModel(){
    console.log('Training...')
    for (let i = 0; i < 100; i++){
      if (i%10==0)
        div.innerHTML += '<b>' + i + '%</b> Complete...<br>';
      const trainingConfiguration = {shuffle: true, epochs: 5}
      const returnValues = await model.fit(
          googStockInput1, 
          googStockOutput, 
          trainingConfiguration
          );
    }
    div.innerHTML += '<b>Done Training!</b></br>';
  }
}