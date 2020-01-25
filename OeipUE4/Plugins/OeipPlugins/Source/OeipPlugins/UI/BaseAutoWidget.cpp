// Fill out your copyright notice in the Description page of Project Settings.

#include "BaseAutoWidget.h"

void UToggleAutoWidget::OnValueChange(bool value) {
	if (onValueChange)
		onValueChange(this, value);
}

void UToggleAutoWidget::Update(bool value) {
	checkBox->SetCheckedState((ECheckBoxState)(value));
}

void UToggleAutoWidget::InitEvent() {
	checkBox->OnCheckStateChanged.AddDynamic(this, &UToggleAutoWidget::OnValueChange);
	ToggleAttribute* atoggle = (ToggleAttribute*)(attribute);
	text->SetText(FText::FromString(atoggle->DisplayName));
	checkBox->CheckedState = ((ECheckBoxState)(atoggle->DefaultValue));
}

void UInputWidget::OnValueChange(FString value) {
	if (onValueChange)
		onValueChange(this, value);
}

void UInputWidget::Update(FString value) {
	textBlock->SetText(FText::FromString(value));
}

void UInputWidget::onTextChange(const FText& rtext) {
	OnValueChange(rtext.ToString());
}

void UInputWidget::InitEvent() {
	textBlock->OnTextChanged.AddDynamic(this, &UInputWidget::onTextChange);
}

void USliderInputWidget::onTextChange(const FText& text) {
	if (bUpdate || !text.IsNumeric())
		return;
	float slider = getSliderValue();
	float value = FCString::Atof(*(text.ToString()));
	if (slider != value)
		setSliderValue(value);
}

void USliderInputWidget::onSliderChange(float value) {
	if (onValueChange)
		onValueChange(this, getSliderValue());
}

float USliderInputWidget::getSliderValue() {
	float fv = slider->GetValue();
	float value = fv * size + offset;
	if (bInt)
		value = (int)value;
	return value;
}

void USliderInputWidget::setSliderValue(float value) {
	if (bAutoRange) {
		offset = value - size * 0.5f;
		slider->SetValue(0.5f);
	}
	else {
		slider->SetValue(FMath::Clamp((value - offset) / size, 0.0f, 1.f));
	}
}

void USliderInputWidget::NativeTick(const FGeometry & MyGeometry, float InDeltaTime) {
	auto play = GetOwningLocalPlayer();
	if (!play)
		return;
	auto cont = play->GetPlayerController(GEngine->GetWorld());
	if (!cont)
		return;
	bool bFocus = textBlock->HasUserFocusedDescendants(cont);
	if (bFocus)
		return;
	bUpdate = true;
	float svalue = getSliderValue();
	if (bInt)
		svalue = (int)svalue;
	FText text = FText::AsNumber(svalue);
	textBlock->SetText(text);
	bUpdate = false;
}

void USliderInputWidget::OnValueChange(float value) {
	if (onValueChange)
		onValueChange(this, value);
}

void USliderInputWidget::Update(float value) {
	setSliderValue(value);
}

void USliderInputWidget::InitEvent() {
	textBlock->OnTextChanged.AddDynamic(this, &USliderInputWidget::onTextChange);
	slider->OnValueChanged.AddDynamic(this, &USliderInputWidget::onSliderChange);
	SliderAttribute* aslider = (SliderAttribute*)(attribute);
	size = aslider->range;
	offset = aslider->offset;
	bInt = aslider->bInt;
	bAutoRange = aslider->bAutoRange;
	//text->SetText(FText::FromString(aslider->DisplayName));
	textBlock->SetText(FText::AsNumber(aslider->DefaultValue));
	slider->SetValue(aslider->DefaultValue);
	if (bInt)
		slider->SetStepSize(1.0f / size);
}

void UDropdownWidget::OnValueChange(int value) {
	if (onValueChange)
		onValueChange(this, value);
}

void UDropdownWidget::Update(int value) {
	if (bAutoAddDefault)
		value = value + 1;
	FString svalue = comboBox->GetOptionAtIndex(value);
	if (svalue.IsEmpty())
		return;
	comboBox->SetSelectedOption(svalue);
}

void UDropdownWidget::onSelectChange(FString SelectedItem, ESelectInfo::Type SelectionType) {
	int index = comboBox->FindOptionIndex(SelectedItem);
	if (bAutoAddDefault) {
		index = index - 1;
	}
	OnValueChange(index);
}

void UDropdownWidget::onSelectParentChange(FString SelectedItem, ESelectInfo::Type SelectionType) {
	if (!dattribut->onFillParentFunc)
		return;
	int index = parent->comboBox->FindOptionIndex(SelectedItem);
	if (parent->bAutoAddDefault) {
		index = index - 1;
	}
	auto options = dattribut->onFillParentFunc(index);
	SetFillOptions(options);
	if (options.Num() > 0)
		Update(0);
}

void UDropdownWidget::InitEvent() {
	comboBox->OnSelectionChanged.AddDynamic(this, &UDropdownWidget::onSelectChange);
	dattribut = (DropdownAttribute*)(attribute);
	bAutoAddDefault = dattribut->bAutoAddDefault;
	if (dattribut->Parent.IsEmpty()) {
		SetFillOptions(dattribut->options);
		Update(dattribut->DefaultValue);
	}
}

void UDropdownWidget::InitParentEvent() {
	if (parent) {
		parent->comboBox->OnSelectionChanged.AddDynamic(this, &UDropdownWidget::onSelectParentChange);
		onSelectParentChange(parent->comboBox->GetSelectedOption(), ESelectInfo::OnMouseClick);
		Update(dattribut->DefaultValue);
	}
}

void UDropdownWidget::SetFillOptions(TArray<FString> options) {
	comboBox->ClearOptions();
	if (bAutoAddDefault)
		comboBox->AddOption("None");
	for (auto option : options) {
		comboBox->AddOption(option);
	}
}
