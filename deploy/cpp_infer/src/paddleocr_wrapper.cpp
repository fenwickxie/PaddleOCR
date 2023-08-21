// PPOCRWrapper.cpp

#include "include/paddleocr_wrapper.h"

namespace PaddleOCR {


	extern "C" void* CreatePPOCRDefaultInstance() {
		return new PPOCR();
	}

	extern "C" void* CreatePPOCRInstance(const bool det, const std::string & det_model_dir, const float det_db_thresh, const float det_db_box_thresh, const float det_db_unclip_ratio, const std::string & det_db_score_mode, const std::string & precision,
		const bool cls, const bool use_angle_cls, const std::string & cls_model_dir, const int cls_batch_num,
		const bool rec, const std::string & rec_model_dir, const int rec_batch_num) {

		return new PPOCR(det, det_model_dir, det_db_thresh, det_db_box_thresh, det_db_unclip_ratio, det_db_score_mode, precision,
			cls, use_angle_cls, cls_model_dir, cls_batch_num,
			rec, rec_model_dir, rec_batch_num);
	}

	extern "C" void DeletePPOCRInstance(void* instance) {
		delete static_cast<PPOCR*>(instance);
	}

	extern "C" void* OCRImage(void* instance, cv::Mat *img) {
		PPOCR* ppocr = static_cast<PPOCR*>(instance);

		// Call ppocr->ocr() and get the result
		std::vector<OCRPredictResult> result = ppocr->ocr(*img);

		// Convert result to a C-friendly structure
		std::vector<OCRPredictResult>* result_array = new std::vector<OCRPredictResult>(result);

		// Return the converted structure as void*
		return static_cast<void*>(result_array);
	}

	extern "C" void* OCRImageBatch(void* instance, cv::Mat * *img_list) {
		PPOCR* ppocr = static_cast<PPOCR*>(instance);

		// Convert img_list to std::vector<cv::Mat>
		std::vector<cv::Mat> images;
		for (int i = 0; img_list[i] != nullptr; ++i) {
			images.push_back(*(img_list[i]));
		}

		// Call ppocr->ocr() and get the result
		std::vector<std::vector<OCRPredictResult>> result = ppocr->ocr(images);

		// Convert result to a C-friendly structure
		std::vector<OCRPredictResult>* result_array = new std::vector<OCRPredictResult>[result.size()];
		for (size_t i = 0; i < result.size(); ++i) {
			result_array[i] = result[i];
		}

		// Return the converted structure as void*
		return static_cast<void*>(result_array);
	}

}

