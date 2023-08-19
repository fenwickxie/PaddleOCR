#pragma once
// PPOCRWrapper.h

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "include/paddleocr.h"

namespace PaddleOCR {
#ifdef __cplusplus
	extern "C" {
#endif

#ifdef PPPOCRWRAPPER_EXPORTS
#define PPPOCRWRAPPER_API __declspec(dllexport)
#else
#define PPPOCRWRAPPER_API __declspec(dllimport)
#endif
		PPPOCRWRAPPER_API void* CreatePPOCRDefaultInstance();
		PPPOCRWRAPPER_API void* CreatePPOCRInstance(const bool det, const std::string& det_model_dir, const float det_db_thresh, const float det_db_box_thresh, const float det_db_unclip_ratio, const std::string& det_db_score_mode, const std::string& precision,
			const bool cls, const bool use_angle_cls, const std::string& cls_model_dir, const int cls_batch_num,
			const bool rec, const std::string& rec_model_dir, const int rec_batch_num);
		PPPOCRWRAPPER_API void DeletePPOCRInstance(void* instance);
		PPPOCRWRAPPER_API void OCRImage(void* instance, cv::Mat* img, std::vector<OCRPredictResult>* results);
		PPPOCRWRAPPER_API void OCRImageBatch(void* instance, std::vector<cv::Mat> *img_list, std::vector<std::vector<OCRPredictResult>>* results);
#ifdef __cplusplus
	}
#endif
}