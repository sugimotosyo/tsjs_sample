/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import {housingDataset, featureDescriptions} from './data';
import * as normalization from './normalization';
import * as ui from './ui';

// Some hyperparameters for model training.
const NUM_EPOCHS = 100; //default was 150
const BATCH_SIZE = 30;
const LEARNING_RATE = 0.01;

const housingData = new housingDataset();
const tensors = {};

// Convert loaded data into tensors and creates normalized versions of the
// features.
export function arraysToTensors() {
  tensors.rawTrainFeatures = tf.tensor2d(housingData.trainFeatures);
  tensors.trainTarget = tf.tensor2d(housingData.trainTarget);
  tensors.rawTestFeatures = tf.tensor2d(housingData.testFeatures);
  tensors.testTarget = tf.tensor2d(housingData.testTarget);
  // Normalize mean and standard deviation of data.
  let {dataMean, dataStd} =
      normalization.determineMeanAndStddev(tensors.rawTrainFeatures);

  tensors.trainFeatures = normalization.normalizeTensor(
      tensors.rawTrainFeatures, dataMean, dataStd);
  tensors.testFeatures =
      normalization.normalizeTensor(tensors.rawTestFeatures, dataMean, dataStd);
};

/**
 * Builds and returns Linear Regression Model.
 *
 * @returns {tf.Sequential} The linear regression model.
 */
export function linearRegressionModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [housingData.numFeatures], units: 1}));

  model.summary();
  return model;
};

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 1 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression model.
 */
export function multiLayerPerceptronRegressionModel1Hidden() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [housingData.numFeatures],
    units: 50,
    activation: 'sigmoid',
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dense({units: 1}));

  model.summary();
  return model;
};

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 2 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression mode  l.
 */
export function multiLayerPerceptronRegressionModel2Hidden() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [housingData.numFeatures],
    units: 50,
    activation: 'sigmoid',
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dense(
      {units: 50, activation: 'sigmoid', kernelInitializer: 'leCunNormal'}));
  model.add(tf.layers.dense({units: 1}));

  model.summary();
  return model;
};


/**
 * Describe the current linear weights for a human to read.
 *
 * @param {Array} kernel Array of floats of length 5.  One value per feature.
 * @returns {List} List of objects, each with a string feature name, and value
 *     feature weight.
 */
export function describeKernelElements(kernel) {
  tf.util.assert(
 
      kernel.length == 5,
      `kernel must be a array of length _ , got ${kernel.length}`);
  const outList = [];
  for (let idx = 0; idx < kernel.length; idx++) {
    outList.push({description: featureDescriptions[idx], value: kernel[idx]});
  }
  return outList;
}

/**
 * Compiles `model` and trains it using the train data and runs model against
 * test data. Issues a callback to update the UI after each epcoh.
 *
 * @param {tf.Sequential} model Model to be trained.
 * @param {boolean} weightsIllustration Whether to print info about the learned
 *  weights.
 */
export async function run(model, modelName, weightsIllustration) {
  model.compile(
      {optimizer: tf.train.sgd(LEARNING_RATE), loss: 'meanSquaredError'});

  let trainLogs = [];
  const container = document.querySelector(`#${modelName} .chart`);

  ui.updateStatus('学習開始...');
  await model.fit(tensors.trainFeatures, tensors.trainTarget, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        await ui.updateModelStatus(
            `エポック ${epoch + 1} / ${NUM_EPOCHS} 終了`, modelName);
        trainLogs.push(logs);
        tfvis.show.history(container, trainLogs, ['loss', 'val_loss'])

        if (weightsIllustration) {
          model.layers[0].getWeights()[0].data().then(kernelAsArr => {
            const weightsList = describeKernelElements(kernelAsArr);
            ui.updateWeightDescription(weightsList);
          });
        }
      }
    }
  });

  ui.updateStatus('学習完了。');
  const result = model.evaluate(
      tensors.testFeatures, tensors.testTarget, {batchSize: BATCH_SIZE});
  const testLoss = result.dataSync()[0];

  const trainLoss = trainLogs[trainLogs.length - 1].loss;
  const valLoss = trainLogs[trainLogs.length - 1].val_loss;
  await ui.updateModelStatus(
      `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
          `Final validation-set loss: ${valLoss.toFixed(4)}\n` +
          `Test-set loss: ${testLoss.toFixed(4)}`,
      modelName);
  
  await evaluateModelOnTestData(model, tensors.testFeatures, tensors.rawTestFeatures, tensors.testTarget);	  

};

export function computeBaseline() {
  const avgPrice = tf.mean(tensors.trainTarget);
  console.log(`平均価格: ${avgPrice.dataSync()}`);
  const baseline = tf.mean(tf.pow(tf.sub(tensors.testTarget, avgPrice), 2));
  console.log(`平均二乗誤差: ${baseline.dataSync()}`);
  const baselineMsg = `平均二乗誤差（誤差の基準として使用）: ${
      baseline.dataSync()[0].toFixed(2)}`;
  ui.updateBaselineStatus(baselineMsg);
};

/**
 * Run inference on some test data.
 *
 * @param model The instance of `tf.Model` to run the inference with.
 * @param xTest Test data feature, a `tf.Tensor` of shape [numTestExamples, 4].
 * @param yTest Test true labels, one-hot encoded, a `tf.Tensor` of shape
 *   [numTestExamples, 3].
 */
async function evaluateModelOnTestData(model, xTest, rawxTest, yTest) {
  ui.clearEvaluateTable();
  var yPredOut = {};
  tf.tidy(() => {
	//テストデータ表示用
    const xData = rawxTest.dataSync();
    const yTrue = yTest.argMax(-1).dataSync();
	
	//予測計算
    const predictOut = model.predict(xTest);
    const yPred = predictOut.dataSync();
	
	//小数点を修正
	var x = 0;
	var len = yPred.length;
	
	while(x < len){ 
		yPredOut[x] = yPred[x].toFixed(2); 
		//console.log(yPredOut[x]);
		x++;
	};
	
    ui.renderEvaluateTable( xData, housingData.testTarget, yPredOut);
	ui.plotData('#data .plot', housingData.testTarget, yPredOut); 
  });
};

document.addEventListener('DOMContentLoaded', async () => {
	
  await housingData.loadData();
  ui.updateStatus('データロード完了、データをテンソルに変換中...');
  arraysToTensors();
  ui.updateStatus(
      'データをテンソルに変換完了。\n' +
      'ボタンを押すと学習を開始します。');
  // TODO Explain what baseline loss is. How it is being computed in this
  // Instance
  ui.updateBaselineStatus('誤差を計算中...');
  computeBaseline();
  await ui.setup();
}, false);
