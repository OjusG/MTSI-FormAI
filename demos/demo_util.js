/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs';
import lengths_trainer from './lengths_trainer.json';
import lengths_user from './lengths_user.json';

var color = 'aqua';
var boundingBoxColor = 'red';
var lineWidth = 2;

export const tryResNetButtonName = 'tryResNetButton';
export const tryResNetButtonText = '[New] Try ResNet50';
const tryResNetButtonTextCss = 'width:100%;text-decoration:underline;';
const tryResNetButtonBackgroundCss = 'background:#e61d5f;';

const allPoints = {
  "nose" : 0,
  "leftEye" : 1,
  "rightEye" : 2,
  "leftEar" : 3,
  "rightEar" : 4,
  "leftShoulder" : 5,
  "rightShoulder" : 6,
  "leftElbow" : 7,
  "rightElbow" : 8,
  "leftWrist" : 9,
  "rightWrist" : 10,
  "leftHip" : 11,
  "rightHip" : 12,
  "leftKnee" : 13,
  "rightKnee" : 14,
  "leftAnkle" : 15,
  "rightAnkle" : 16
}

var avgLengths = {
  "torso_top": [5,6,0],
  "torso_bottom": [11,12,0],
  "torso_right": [6,12,0],
  "torso_left": [5,11,0],
  "left_upper_arm": [5,7,0],
  "right_upper_arm": [6,8,0],
  "left_forearm": [7,9,0],
  "right_forearm": [8,10,0],
  "left_thigh": [11,13,0],
  "right_thigh": [12,14,0],
  "left_calf": [13,15,0],
  "right_calf": [14,16,0]
}

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export function isMobile() {
  return isAndroid() || isiOS();
}

function setDatGuiPropertyCss(propertyText, liCssString, spanCssString = '') {
  var spans = document.getElementsByClassName('property-name');
  for (var i = 0; i < spans.length; i++) {
    var text = spans[i].textContent || spans[i].innerText;
    if (text == propertyText) {
      spans[i].parentNode.parentNode.style = liCssString;
      if (spanCssString !== '') {
        spans[i].style = spanCssString;
      }
    }
  }
}

export function updateTryResNetButtonDatGuiCss() {
  setDatGuiPropertyCss(
      tryResNetButtonText, tryResNetButtonBackgroundCss,
      tryResNetButtonTextCss);
}

/**
 * Toggles between the loading UI and the main canvas UI.
 */
export function toggleLoadingUI(
    showLoadingUI, loadingDivId = 'loading', mainDivId = 'main') {
  if (showLoadingUI) {
    document.getElementById(loadingDivId).style.display = 'block';
    document.getElementById(mainDivId).style.display = 'none';
  } else {
    document.getElementById(loadingDivId).style.display = 'none';
    document.getElementById(mainDivId).style.display = 'block';
  }
}

function toTuple({y, x}) {
  return [y, x];
}

export function drawPoint(ctx, y, x, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

/**
 * Draws a line on a canvas, i.e. a joint
 */
export function drawSegment([ay, ax], [by, bx], color, scale, ctx, person=undefined) {
  ctx.beginPath();
  ctx.moveTo(ax * scale, ay * scale);
  ctx.lineTo(bx * scale, by * scale);
  if (person) {
    ctx.setLineDash([10, 3]);
    ctx.restore();
  } else {
    ctx.setLineDash([0]);
  }
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.stroke();

}

/**
 * Draws a pose skeleton by looking up all adjacent keypoints/joints
 */
export function drawSkeleton(keypoints, minConfidence, ctx, person=undefined, scale = 1) {

  const adjacentKeyPoints =
      posenet.getAdjacentKeyPoints(keypoints, minConfidence);

  adjacentKeyPoints.forEach((keypoints) => {
    // if (keypoints[0].part.includes("Knee") || keypoints[0].part.includes("Ankle")) {
    //   return;
    // }
    drawSegment(
        toTuple(keypoints[0].position), toTuple(keypoints[1].position), color,
        scale, ctx, person);
  });
}

/**
 * Draws a pose skeleton by looking up all adjacent keypoints/joints
 */
export function drawTrainerSkeleton(keypoints, minConfidence, ctx, person=undefined ) {
  color = 'red';
  lineWidth = 4;
  drawSkeleton(keypoints, minConfidence, ctx, person );
  color = 'yellow';
  lineWidth = 3;
}

/**
 * Draw pose keypoints onto a canvas
 */
export function drawKeypoints(keypoints, minConfidence, ctx, scale = 1) {
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];

    if (keypoint.score < minConfidence) {
      continue;
    }

    const {y, x} = keypoint.position;
    drawPoint(ctx, y * scale, x * scale, 3, color);
  }
}

/**
 * Draw the bounding box of a pose. For example, for a whole person standing
 * in an image, the bounding box will begin at the nose and extend to one of
 * ankles
 */
export function drawBoundingBox(keypoints, ctx) {
  const boundingBox = posenet.getBoundingBox(keypoints);

  ctx.rect(
      boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX,
      boundingBox.maxY - boundingBox.minY);

  ctx.strokeStyle = boundingBoxColor;
  ctx.stroke();
}

/**
 * Converts an arary of pixel data into an ImageData object
 */
export async function renderToCanvas(a, ctx) {
  const [height, width] = a.shape;
  const imageData = new ImageData(width, height);

  const data = await a.data();

  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * 3;

    imageData.data[j + 0] = data[k + 0];
    imageData.data[j + 1] = data[k + 1];
    imageData.data[j + 2] = data[k + 2];
    imageData.data[j + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

/**
 * Draw an image on a canvas
 */
export function renderImageToCanvas(image, size, canvas) {
  canvas.width = size[0];
  canvas.height = size[1];
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);
}

/**
 * Draw heatmap values, one of the model outputs, on to the canvas
 * Read our blog post for a description of PoseNet's heatmap outputs
 * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
 */
export function drawHeatMapValues(heatMapValues, outputStride, canvas) {
  const ctx = canvas.getContext('2d');
  const radius = 5;
  const scaledValues = heatMapValues.mul(tf.scalar(outputStride, 'int32'));

  drawPoints(ctx, scaledValues, radius, color);
}

/**
 * Used by the drawHeatMapValues method to draw heatmap points on to
 * the canvas
 */
function drawPoints(ctx, points, radius, color) {
  const data = points.buffer().values;

  for (let i = 0; i < data.length; i += 2) {
    const pointY = data[i];
    const pointX = data[i + 1];

    if (pointX !== 0 && pointY !== 0) {
      ctx.beginPath();
      ctx.arc(pointX, pointY, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }
}

/**
 * Draw offset vector values, one of the model outputs, on to the canvas
 * Read our blog post for a description of PoseNet's offset vector outputs
 * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
 */
export function drawOffsetVectors(
    heatMapValues, offsets, outputStride, scale = 1, ctx) {
  const offsetPoints =
      posenet.singlePose.getOffsetPoints(heatMapValues, outputStride, offsets);

  const heatmapData = heatMapValues.buffer().values;
  const offsetPointsData = offsetPoints.buffer().values;

  for (let i = 0; i < heatmapData.length; i += 2) {
    const heatmapY = heatmapData[i] * outputStride;
    const heatmapX = heatmapData[i + 1] * outputStride;
    const offsetPointY = offsetPointsData[i];
    const offsetPointX = offsetPointsData[i + 1];

    drawSegment(
        [heatmapY, heatmapX], [offsetPointY, offsetPointX], color, scale, ctx);
  }
}

function distanceFormula(point1, point2) {
  return Math.sqrt(Math.pow(point2.position.y - point1.position.y, 2) + Math.pow(point2.position.x - point1.position.x, 2));
}
/*
"torso_top": 0,
"torso_bottom": 0,
"torso_right": 0,
"torso_left": 0,
"left_upper_arm": 0,
"right_upper_arm": 0,
"left_forearm": 0,
"right_forearm": 0,
"left_thigh": 0,
"right_thigh": 0,
"left_calf": 0,
"right_calf": 0
*/
export function calculateLengths(all_poses) {
  for (var frame_name in all_poses) {
    var frame_num = parseInt(frame_name.split('_')[1]);

    for (var segment in avgLengths) {
      var ind1 = avgLengths[segment][0];
      var ind2 = avgLengths[segment][1];
      var currAvg = avgLengths[segment][2];

      avgLengths[segment][2] = (currAvg * frame_num + distanceFormula(all_poses[frame_name][ind1], all_poses[frame_name][ind2])) / (frame_num + 1);;
    }
  }
  return avgLengths;
}

export function transformKeypoints(keypoints, trainerRefPoint) {
  if (typeof(trainerRefPoint) != "undefined" && typeof(referencePoint) != "undefined") {
    var scaleFactor = lengths_user.right_upper_arm[2] / lengths_trainer.right_upper_arm[2];
    trainerRefPoint.position.x *= scaleFactor;
    trainerRefPoint.position.y *= scaleFactor;
    var shiftX = referencePoint.x - trainerRefPoint.position.x;
    var shiftY = referencePoint.y - trainerRefPoint.position.y;

    for (var i=0;i<keypoints.length;i++) {
      if (keypoints[i].part != "rightShoulder") {
        keypoints[i].position.x *= scaleFactor;
        keypoints[i].position.y *= scaleFactor;
      }
      keypoints[i].position.x += shiftX;
      keypoints[i].position.y += shiftY;
    }
  }

  return keypoints;
}
