#pragma once
#include "OeipCommon.h"
#include <vector>
#include <mutex>

struct Ring
{
	int32_t read;
	int32_t write;
	int32_t size;
	bool bFull;
};

#define DEFAULTRBUFFERSIZE 60*1024

class OEIPDLL_EXPORT RingBuffer
{
private:
	std::mutex mtx;
	Ring ring = {};
	std::vector<uint8_t> data;
public:
	RingBuffer() :RingBuffer(DEFAULTRBUFFERSIZE) {};
	RingBuffer(int32_t size) {
		ring.size = size;
		ring.read = 0;
		ring.write = 0;
		ring.bFull = false;
		resize(size);
	}
	~RingBuffer() {
		data.clear();
	}
public:
	void resize(int32_t size) {
		ring.size = size;
		data.resize(size);
	}

	void clear() {
		ring.read = 0;
		ring.write = 0;
		ring.bFull = 0;
	}
	//多少可写
	int32_t sizeWrite() {
		if (bFull())
			return 0;
		if (ring.write < ring.read) {
			return ring.read - ring.write;
		}
		else {
			return (ring.size - ring.write) + ring.read;
		}
	}
	//多少可读
	int32_t sizeRead() {
		if (ring.write >= ring.read && !ring.bFull) {
			return ring.write - ring.read;
		}
		else {
			return ring.size - ring.read + ring.write;
		}
	}

	bool bEmpty() {
		return (ring.read == ring.write) && !ring.bFull;
	}

	bool bFull() {
		return (ring.read == ring.write) && ring.bFull;
	}

	//写数据
	int32_t push(uint8_t* pdata, int32_t len) {
		std::unique_lock <std::mutex> lck(mtx);
		if (bFull()) {
			logMessage(OEIP_WARN, "push ring buffer is full.");
			return -1;
		}
		if (sizeWrite() < len) {
			logMessage(OEIP_WARN, "push ring buffer is maxout.");
			return -1;
		}
		if (ring.write < ring.read || ring.size - ring.write >= len) {
			memcpy(data.data() + ring.write, pdata, len);
			ring.write += len;
		}
		else {
			int32_t wend = ring.size - ring.write;
			memcpy(data.data() + ring.write, pdata, wend);
			memcpy(data.data(), pdata + wend, len - wend);
			ring.write = len - wend;
		}
		ring.write = ring.write % ring.size;
		if (ring.write == ring.read) {
			ring.bFull = true;
		}
		return 0;
	}

	//读数据
	int32_t pull(uint8_t* pdata, int32_t len) {
		std::unique_lock <std::mutex> lck(mtx);
		if (sizeRead() < len) {
			//logMessage(zmf_warn, "pull ring buffer is maxout.");
			return -1;
		}
		if (ring.write > ring.read || ring.size - ring.write >= len) {
			memcpy(pdata, data.data() + ring.read, len);
			ring.read += len;
		}
		else {
			int rend = ring.size - ring.read;
			memcpy(pdata, data.data() + ring.read, rend);
			memcpy(pdata + rend, data.data(), len - rend);
			ring.read = len - rend;
		}
		ring.read = ring.read % ring.size;
		ring.bFull = false;
		return 0;
	}
};

