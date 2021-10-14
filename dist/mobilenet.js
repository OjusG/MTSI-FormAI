"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var base_model_1 = require("./base_model");
var MobileNet = (function (_super) {
    __extends(MobileNet, _super);
    function MobileNet() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    MobileNet.prototype.preprocessInput = function (input) {
        return tf.tidy(function () { return tf.div(input, 127.5).sub(1.0); });
    };
    MobileNet.prototype.nameOutputResults = function (results) {
        var offsets = results[0], heatmap = results[1], displacementFwd = results[2], displacementBwd = results[3];
        return { offsets: offsets, heatmap: heatmap, displacementFwd: displacementFwd, displacementBwd: displacementBwd };
    };
    return MobileNet;
}(base_model_1.BaseModel));
exports.MobileNet = MobileNet;
//# sourceMappingURL=mobilenet.js.map