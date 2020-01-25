// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include <functional>
#include "UObject/EnumProperty.h"

struct BaseAttribute
{
protected:
	int index = -1;
public:
	FString MemberName;
	FString DisplayName;
public:
	virtual int GetIndex() { return index; };
	virtual ~BaseAttribute() {};
};

struct ToggleAttribute : public BaseAttribute
{
public:
	ToggleAttribute() { index = 0; }
public:
	bool DefaultValue = false;
};

struct InputeAttribute : public BaseAttribute
{
public:
	InputeAttribute() { index = 1; }
public:
	FString DefaultValue = "";
};

struct SliderAttribute : public BaseAttribute
{
public:
	SliderAttribute() { index = 2; }
public:
	float DefaultValue = 0.0f;
	float range = 1.0f;
	//范围的起始值
	float offset = 0.0f;
	bool bAutoRange = false;
	bool bInt = false;
};

struct DropdownAttribute : public BaseAttribute
{
public:
	DropdownAttribute() { index = 3; }
public:
	int DefaultValue = 0;
	FString Parent;
	bool bAutoAddDefault = false;
	std::function<TArray<FString>(int)> onFillParentFunc;
	TArray<FString> options;
public:
	void InitOptions(bool bAuto, TArray<FString> optionArray)
	{
		bAutoAddDefault = bAuto;
		options = optionArray;
	};
	void InitOptions(bool bAuto, FString parent, std::function<TArray<FString>(int)> onFunc)
	{
		bAutoAddDefault = bAuto;
		Parent = parent;
		onFillParentFunc = onFunc;
	}
};

