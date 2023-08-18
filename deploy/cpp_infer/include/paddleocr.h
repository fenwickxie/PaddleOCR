// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <include/ocr_cls.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>

namespace PaddleOCR {

extern "C" class  __declspec(dllexport) PPOCR {
public:
  //explicit PPOCR();
  explicit PPOCR(const bool det = true, const std::string& det_model_dir = "./inference/det_infer", const float det_db_thresh = 0.3, const float det_db_box_thresh = 0.6, const float det_db_unclip_ratio = 1.8, const std::string& det_db_score_mode = "fast", const std::string& precision = "fp16",
      const bool cls = true, const bool use_angle_cls = true, const std::string& cls_model_dir = "./inference/cls_infer", const int cls_batch_num = 1,
      const bool rec = true, const std::string& rec_model_dir = "./inference/rec_infer", const int rec_batch_num = 6);
  ~PPOCR() = default;

  std::vector<std::vector<OCRPredictResult>> ocr(std::vector<cv::Mat> img_list,
                                                 bool det = true,
                                                 bool rec = true,
                                                 bool cls = true);
  std::vector<OCRPredictResult> ocr(cv::Mat img, bool det = true,
                                    bool rec = true, bool cls = true);

  void reset_timer();
  void benchmark_log(int img_num);

protected:
  std::vector<double> time_info_det = {0, 0, 0};
  std::vector<double> time_info_rec = {0, 0, 0};
  std::vector<double> time_info_cls = {0, 0, 0};

  void det(cv::Mat img, std::vector<OCRPredictResult> &ocr_results);
  void rec(std::vector<cv::Mat> img_list,
           std::vector<OCRPredictResult> &ocr_results);
  void cls(std::vector<cv::Mat> img_list,
           std::vector<OCRPredictResult> &ocr_results);

private:
  std::unique_ptr<DBDetector> detector_;
  std::unique_ptr<Classifier> classifier_;
  std::unique_ptr<CRNNRecognizer> recognizer_;
};
extern "C" __declspec(dllexport) PPOCR * CreatePPOCR();
extern "C" __declspec(dllexport) void DeletePPOCR(PPOCR * obj);
} // namespace PaddleOCR
