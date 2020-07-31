#pragma once
#include <iostream>
namespace Templatetest
{
	template<int index, typename... Types>
	struct At;

	template<int index, typename First, typename... Types>
	struct At<index, First, Types...>
	{
		using ptype = typename At<index - 1, Types...>::ptype;
	};

	template<typename T, typename... Types>
	struct At<0, T, Types...>
	{
		using ptype = T;
	};

	using x = At<0, int, double, char>::ptype;

	template<int32_t index>
	class Fruit
	{
		using T2 = At<0, int, double, char>::ptype;
	public:
		T2 t = {};
	};

	template<>
	class Fruit<0> {};

	template<int32_t LayerType>
	class TemplateLayer
	{
		int index = 0;
	public:
		void update(void* index) {
			Fruit<0> xxx;
			//auto xx = xxx.t;
			using xxxx = typename At<LayerType, int, double, char>::ptype;
			//std::cout << typeid(*index) << std::endl;
			xxxx* x = (xxxx*)index;
			std::cout << *x << std::endl;
		}
	};
}