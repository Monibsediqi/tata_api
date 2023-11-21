#include "TataBase.h"
//#include "easylogging++.h"
#include <experimental/filesystem>
#include<c10/cuda/CUDACachingAllocator.h>

#define PRINT(x) cout << x << endl;
#define INFO(x) cout << "INFO: " << x << endl;
#define ERR(x) cerr << "ERROR: " << x << endl;


namespace fs = std::experimental::filesystem;

using namespace tata;

TataBase::TataBase(const Configuration& config) : m_config(config)
{
};
void TataBase::clearCache() {
	c10::cuda::CUDACachingAllocator::emptyCache();
}
void TataBase::clearThread() {
	lock_guard<mutex> guard(_checkMutex);
	if (_check.st_type == StatusType::IDLE || _check.st_type == StatusType::DONE || _check.st_type == StatusType::STOPPED) {
		if (_thread.size() > 0) {
			if (_thread[0].joinable())
				_thread[0].join();
			_thread.clear();
		}
		_check.st_type = StatusType::IDLE;
		_check.progress = 0;
		_check.exception = "";
		_check.ex_type = ExceptionType::OCCURED;
	}
}
bool TataBase::checkStop() {
	lock_guard<mutex> guard(_checkMutex);
	if (_check.st_type == StatusType::STOPPED) {
		_check.st_type = StatusType::STOPPING;
		_check.progress = 0;
		_check.exception = "";
		_check.ex_type = ExceptionType::OCCURED;
		return true;
	}
	return false;
}
bool TataBase::checkStopSetProgress(int start, int end, float progress) {
	//progress from 0.0 to 1.0
	//possible status to enter here: NORMAL, STOPPING
	lock_guard<mutex> guard(_checkMutex);
	_check.progress = start + progress * (end - start);
	if (_check.progress > end)
		_check.progress = end;
	if (_check.st_type == StatusType::STOPPING) {
		_check.st_type = StatusType::STOPPED;
		return true;
	}
	return false;

}
bool TataBase::tiSepStop() {
	bool toStop = false;
	{
		lock_guard<mutex> guard(_checkMutex);
		if (_check.st_type == StatusType::NORMAL || _check.st_type == StatusType::IDLE) {
			toStop = true;
			_check.st_type = StatusType::STOPPING;
			INFO("Send stop message\n");
		}
	}
	if (toStop) {
		//wait until thread done
		INFO ("Wait\n");
		if (_thread.size() > 0) {
			if (_thread[0].joinable())
				_thread[0].join();
			_thread.erase(_thread.begin() + 0);
		}
		{
			lock_guard<mutex> guard(_checkMutex);
			if (_check.st_type == StatusType::STOPPING)
				_check.st_type = StatusType::STOPPED;
		}
		INFO("tiSepStop() Stopped\n");
	}
	return toStop;
}
TataBase::~TataBase()
{
	//cleanup: make sure the thread is shutdown
	if (_thread.size() > 0) {
		if (_thread[0].joinable())
			_thread[0].join();
		_thread.clear();
	}
};