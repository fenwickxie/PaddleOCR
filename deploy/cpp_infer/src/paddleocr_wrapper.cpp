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

	extern "C" void OCRImage(void* instance, cv::Mat * img, std::vector<OCRPredictResult> *results) {
		PPOCR* ppocr = static_cast<PPOCR*>(instance);
		std::vector<OCRPredictResult> ocrResults = ppocr->ocr(*img);
	}
	extern "C" void OCRImageBatch(void* instance, std::vector<cv::Mat> *img_list, std::vector<std::vector<OCRPredictResult>> *results) {
		PPOCR* ppocr = static_cast<PPOCR*>(instance);
		std::vector<std::vector<OCRPredictResult>> ocrResults = ppocr->ocr(*img_list);
	}
}

