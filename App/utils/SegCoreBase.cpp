/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.sediqi@medicalip.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/

#include "../SegCoreBase.h"


SegCoreBase::SegCoreBase()
{

};
SegCoreBase::~SegCoreBase()
{

};
void SegCoreBase::setProduct(const Product& product)
{
	m_product = product;
}
void SegCoreBase::setAITask(const AITask& ai_task)
{
	m_ai_task = ai_task;
}
void SegCoreBase::setBucket(Bucket& in_bucket)
{
	m_bucket = in_bucket;
}
Product SegCoreBase::getProduct() const {
	return m_product;
}
AITask SegCoreBase::getAITask() const {
	return m_ai_task;
}
Bucket SegCoreBase::getBucket() const {
	return m_bucket;
}
//OutBucket SegCoreBase::getOutBucket() const
//{
//	return m_out_bucket;
//}
void SegCoreBase::run() {
	// do something in here
}
void SegCoreBase::setNormMethod() {
	// do something in here
}
//Tensor SegCoreBase::normalize(Tensor& t_data, const NormMethod) {
//	torch::Tensor t_norm_data;
//
//	if (m_norm_method == NormMethod::MINMAX) {
//		//do min max normalization in accordance to python 
//		INFO("Using MinMax normalization: ");
//		torch::Tensor min = t_data.min();
//		torch::Tensor max = t_data.max();
//		t_norm_data = (t_data - min) / (max - min);
//		return t_norm_data;
//	}
//	else if (m_norm_method == NormMethod::ZSCORE) {
//		INFO("Using ZSCORE normalization: ");
//		torch::Tensor mean = t_data.mean();
//		torch::Tensor std = t_data.std();
//
//		t_norm_data = (t_data - mean) / std;
//		return t_norm_data;
//	}
//	else if (m_norm_method == NormMethod::PERCENTILE) {
//		float eps = 1e-10;
//
//		auto t_mean = t_data.mean();
//		auto t_std = t_data.std();
//		//auto tensor_neg2std = torch::where(tensor < t_mean - (2 * t_std), t_mean - (2 * t_std), tensor);
//
//		auto percentile0 = calculatePercentile(t_data, 0);
//		auto percentile99 = calculatePercentile(t_data, 99);
//
//		auto normalized_a = (t_data - percentile0) / ((percentile99 - percentile0) + eps);
//
//		//PRINT("normalized_a" << normalized_a);
//
//		return normalized_a;
//	}
//
//	else if (m_norm_method == NormMethod::NONE) {
//		// do nothing 
//		INFO("Using NONE normalization: ");
//		return t_data;
//	}
//
//	else {
//		ERR("Error: Invalid normalization method. ");
//		return t_data;
//	}
//}
//float SegCoreBase::calculatePercentile(const Tensor& tensor, float percentile)  {
//	auto sorted_data = torch::sort(tensor.flatten());
//	Tensor sorted_values = std::get<0>(sorted_data);
//	Tensor sorted_indices = std::get<1>(sorted_data);
//	int index = (percentile / 100.0) * sorted_values.size(0);
//	return sorted_values[index].item<float>();
//}
