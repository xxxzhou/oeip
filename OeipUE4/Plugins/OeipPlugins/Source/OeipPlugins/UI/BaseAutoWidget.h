// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UMG.h"
#include "Blueprint/UserWidget.h"
#include "BaseAttribute.h"
#include "BaseAutoWidget.generated.h"

template<typename T>
class ComponentTemplate
{
public:
	typedef T Type;
	//DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnValueChange, ComponentTemplate<T>*, compent, T, t);
	//TBaseDelegate<void, ComponentTemplate<T>*, T> onValueChange;
	//FOnValueChange onValueChange;
	typedef ComponentTemplate<T> WidgetT;
public:
	ComponentTemplate() {};
	virtual ~ComponentTemplate() {};
public:
	//UI显示名称
	FString DisplayName;
	//UI更新后回调
	std::function<void(ComponentTemplate<T>*, T)> onValueChange;
	//UI对应描述，初始化UI时使用
	BaseAttribute* attribute;
	//UI对应的元数据
	UProperty* uproperty = nullptr;
public:
	//UI更新后统一调用事件onValueChange
	virtual void OnValueChange(T value) {};
	//数据更新后去调用UI刷新
	virtual void Update(T value) {};
};

UCLASS()
class UBaseAutoWidget : public UUserWidget
{
	GENERATED_BODY()
public:
	UBaseAutoWidget(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer) {};
public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = AutoWidget)
		UTextBlock * text;
	int index = -1;
	FString memberName;
public:
	//用于蓝图子类赋值UI给C++中
	UFUNCTION(BlueprintNativeEvent)
		void InitWidget();
	virtual void InitWidget_Implementation() {};
	//初始化UI事件与默认值
	virtual void InitEvent() { };
};

UCLASS()
class UToggleAutoWidget : public UBaseAutoWidget, public ComponentTemplate<bool>
{
	GENERATED_BODY()
public:
	UToggleAutoWidget(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer) {};
public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = AutoWidget)
		UCheckBox * checkBox;
public:
	UFUNCTION(BlueprintCallable)
		virtual	void OnValueChange(bool value) override;
	virtual void Update(bool value) override;
public:
	virtual void InitEvent() override;
};

UCLASS()
class UInputWidget : public UBaseAutoWidget, public ComponentTemplate<FString>
{
	GENERATED_BODY()
public:
	UInputWidget(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer) {};
public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = AutoWidget)
		UEditableTextBox * textBlock;
public:
	UFUNCTION(BlueprintCallable)
		virtual	void OnValueChange(FString value) override;
	virtual void Update(FString value) override;
private:
	UFUNCTION(BlueprintCallable)
		void onTextChange(const FText& rtext);
public:
	virtual void InitEvent() override;
};

UCLASS()
class USliderInputWidget : public UBaseAutoWidget, public ComponentTemplate<float>
{
	GENERATED_BODY()
public:
	USliderInputWidget(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer) {};
public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = AutoWidget)
		USlider * slider;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = AutoWidget)
		UEditableTextBox * textBlock;
private:
	//大小
	float size = 1.0f;
	//起始值
	float offset = 0.0f;
	//textbox是否正在编辑
	bool bUpdate = false;
	//是否自动选择范围
	bool bAutoRange = false;
	//是否整形
	bool bInt = false;
private:
	UFUNCTION(BlueprintCallable)
		void onTextChange(const FText& rtext);
	UFUNCTION(BlueprintCallable)
		void onSliderChange(float value);
	float getSliderValue();
	void setSliderValue(float value);
protected:
	virtual void NativeTick(const FGeometry& MyGeometry, float InDeltaTime) override;
public:
	UFUNCTION(BlueprintCallable)
		virtual	void OnValueChange(float value) override;
	virtual void Update(float value) override;
public:
	virtual void InitEvent() override;
};

UCLASS()
class UDropdownWidget : public UBaseAutoWidget, public ComponentTemplate<int>
{
	GENERATED_BODY()
public:
	UDropdownWidget(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer) {};
public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = AutoWidget)
		UComboBoxString * comboBox;
	bool bAutoAddDefault = false;
	UDropdownWidget* parent;
	DropdownAttribute* dattribut;
public:
	UFUNCTION(BlueprintCallable)
		virtual	void OnValueChange(int value) override;
	virtual void Update(int value) override;
private:
	UFUNCTION(BlueprintCallable)
		void onSelectChange(FString SelectedItem, ESelectInfo::Type SelectionType);
	UFUNCTION(BlueprintCallable)
		void onSelectParentChange(FString SelectedItem, ESelectInfo::Type SelectionType);
public:
	virtual void InitEvent() override;
	void InitParentEvent();
	void SetFillOptions(TArray<FString> options);
};
