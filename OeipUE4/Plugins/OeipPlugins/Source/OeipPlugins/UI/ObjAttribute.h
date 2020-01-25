// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"

#include "CoreMinimal.h"
#include "BaseAttribute.h"
#include "UMG.h"
#include "BaseAutoWidget.h"

using namespace std;
using namespace std::placeholders;

template<class U>
class ObjAttribute
{
private:
	//绑定的对象
	U * obj = nullptr;
	//绑定对象的元结构
	UStruct* structDefinition = nullptr;
	//生成UI的box
	UVerticalBox* panel = nullptr;
	//绑定对象元数据描述
	TArray<BaseAttribute*> attributeList;
	//根据元数据描述生成的UI
	TArray<UBaseAutoWidget*> widgetList;
	//是否绑定
	bool bBind = false;
	//当绑定对象改变后引发的回调
	std::function<void(ObjAttribute<U>*, FString name)> onObjChangeHandle = nullptr;
public:
	//绑定一个对象到UVerticalBox上，根据arrtList与对象的元数据以及对应UI模版生成UI在box上,对应UI的变动会自动更新到对象内存数据中
	void Bind(U* pobj, UVerticalBox* box, TArray<BaseAttribute*> arrtList, TArray<UClass*> templateWidgetList, UWorld* world)
	{
		if (bBind)
			return;
		obj = pobj;
		structDefinition = U::StaticStruct();
		attributeList = arrtList;
		panel = box;
		for (auto attribute : attributeList)
		{
			int index = attribute->GetIndex();
			//TSubclassOf<UToggleAutoWidget> togglewidget = LoadClass<UToggleAutoWidget>(nullptr, TEXT("WidgetBlueprint'/Game/UI/togglePanel.togglePanel_C'"));
			auto templateWidget = templateWidgetList[index];
			if (!templateWidget)
				continue;
			if (index == 0)
				InitWidget<UToggleAutoWidget>(attribute, templateWidget, world);
			else if (index == 1)
				InitWidget<UInputWidget>(attribute, templateWidget, world);
			else if (index == 2)
				InitWidget<USliderInputWidget>(attribute, templateWidget, world);
			else if (index == 3)
				InitWidget<UDropdownWidget>(attribute, templateWidget, world);
		}
		for (auto widget : widgetList)
		{
			panel->AddChild(widget);
		}
		UpdateDropdownParent();
		bBind = true;
	};

	//1 probj = null如直接更新内存数据，然后反馈给UI
	//2 probj != null绑定的同类型别的对象,用这对象更新UI,并且后续UI更新反馈给此新对象
	void Update(U* pobj = nullptr)
	{
		if (!bBind)
			return;
		if (pobj)
			obj = pobj;
		for (auto widget : widgetList)
		{
			int index = widget->index;			
			if (index == 0)
				GetValue<UToggleAutoWidget>(widget);
			else if (index == 1)
				GetValue<UInputWidget>(widget);
			else if (index == 2)
				GetValue<USliderInputWidget>(widget);
			else if (index == 3)
				GetValue<UDropdownWidget>(widget);
		}
	}

	//当对应结构的UI改动更新对象后，返回的回调，第一个参数表示当前对象，第二个参数表示对应字段名
	void SetOnObjChangeAction(std::function<void(ObjAttribute<U>*, FString name)> onChange)
	{
		onObjChangeHandle = onChange;
	}

	//返回当前Bind的对象
	U* GetObj()
	{
		return obj;
	}

	//根据字段名得到对应UToggleAutoWidget/UInputWidget/USliderInputWidget/UDropdownWidget
	template<typename A>
	A* GetWidget(FString name)
	{
		for (auto widget : widgetList)
		{
			A* awidget = dynamic_cast<A*>(widget);
			if (awidget && awidget->memberName == name)
			{
				return awidget;
			}
		}
		return nullptr;
	}

private:
	template<typename A>
	void InitWidget(BaseAttribute* attribute, UClass* widgetTemplate, UWorld* world)
	{
		auto widget = CreateWidget<A>(world, widgetTemplate);
		widget->onValueChange = std::bind(&ObjAttribute<U>::SetValue<A::Type>, this, _1, _2);
		widget->attribute = attribute;
		widget->uproperty = FindProperty(attribute->MemberName);
		widget->index = attribute->GetIndex();
		//调用对应蓝图的UI赋值
		widget->InitWidget();
		//关联UI的事件到如上的onValueChange中
		widget->InitEvent();
		widget->memberName = attribute->MemberName;
		widget->text->SetText(FText::FromString(attribute->DisplayName));
		widgetList.Add(widget);
	}

	//在对应的Widget上直接保存此UProperty对象，此后更新数据/UI更快
	UProperty* FindProperty(FString name)
	{
		for (TFieldIterator<UProperty> It(structDefinition); It; ++It)
		{
			UProperty* Property = *It;
			if (Property->GetName() == name)
			{
				return Property;
			}
		}
		return nullptr;
	}

	//当对应的UI改动后，UI影响对应obj的值，泛型t表示对应UI返回的数据
	//ComponentTemplate对应的泛型t是固定的，但是数据结构里的字段类型可多种，转化逻辑在如下写好就行
	template<typename T>
	void SetValue(ComponentTemplate<T>* widget, T t)
	{
		if (widget->uproperty != nullptr)
		{
			ValueToUProperty(widget->uproperty, t);
			if (onObjChangeHandle)
			{
				onObjChangeHandle(this, widget->uproperty->GetName());
			}
		}
	};

	void ValueToUProperty(UProperty* Property, bool t)
	{
		void* Value = Property->ContainerPtrToValuePtr<uint8>(obj);
		if (UBoolProperty *BoolProperty = Cast<UBoolProperty>(Property))
		{
			BoolProperty->SetPropertyValue(Value, t);
		}
	};

	void ValueToUProperty(UProperty* Property, float t)
	{
		void* Value = Property->ContainerPtrToValuePtr<uint8>(obj);
		if (UNumericProperty *NumericProperty = Cast<UNumericProperty>(Property))
		{
			if (NumericProperty->IsFloatingPoint())
			{
				NumericProperty->SetFloatingPointPropertyValue(Value, (float)t);
			}
			else if (NumericProperty->IsInteger())
			{
				NumericProperty->SetIntPropertyValue(Value, (int64)t);
			}
		}
	};

	void ValueToUProperty(UProperty* Property, FString t)
	{
		void* Value = Property->ContainerPtrToValuePtr<uint8>(obj);
		if (UStrProperty *StringProperty = Cast<UStrProperty>(Property))
		{
			StringProperty->SetPropertyValue(Value, t);
		}
	}

	void ValueToUProperty(UProperty* Property, int t)
	{
		void* Value = Property->ContainerPtrToValuePtr<uint8>(obj);
		if (UNumericProperty *NumericProperty = Cast<UNumericProperty>(Property))
		{
			if (NumericProperty->IsFloatingPoint())
			{
				NumericProperty->SetFloatingPointPropertyValue(Value, (int64)t);
			}
			else if (NumericProperty->IsInteger())
			{
				NumericProperty->SetIntPropertyValue(Value, (int64)t);
			}
		}
		else if (UEnumProperty* EnumProperty = Cast<UEnumProperty>(Property))
		{
			EnumProperty->GetUnderlyingProperty()->SetIntPropertyValue(Value, (int64)t);
		}
	}

	//从对应的obj里去取值更新UI，会转到ComponentTemplate::Update
	//同SetValue，ComponentTemplate类型固定，数据结构类型可多种，多种需要写相应的转化逻辑
	template<typename A>//template<typename T, typename A>
	void GetValue(UBaseAutoWidget* baseWidget)//ComponentTemplate<T>* widget, T* t)
	{
		A* widget = (A*)baseWidget;
		if (widget->uproperty != nullptr)
		{
			A::Type t;
			if (UPropertyToValue(widget->uproperty, t))
				widget->Update(t);
		}
		//A* widget = (A*)baseWidget;
		//for (TFieldIterator<UProperty> It(structDefinition); It; ++It)
		//{
		//	UProperty* Property = *It;
		//	FString PropertyName = Property->GetName();
		//	if (PropertyName == widget->attribute->MemberName)
		//	{
		//		A::Type t;
		//		if (UPropertyToValue(Property, t))
		//			widget->Update(t);
		//	}
		//}
	};

	bool UPropertyToValue(UProperty* Property, bool& t)
	{
		void* Value = Property->ContainerPtrToValuePtr<uint8>(obj);
		if (UBoolProperty *BoolProperty = Cast<UBoolProperty>(Property))
		{
			bool value = BoolProperty->GetPropertyValue(Value);
			t = value;
			return true;
		}
		return false;
	};

	bool UPropertyToValue(UProperty* Property, float& t)
	{
		void* Value = Property->ContainerPtrToValuePtr<uint8>(obj);
		if (UNumericProperty *NumericProperty = Cast<UNumericProperty>(Property))
		{
			if (NumericProperty->IsFloatingPoint())
			{
				float value = NumericProperty->GetFloatingPointPropertyValue(Value);
				t = value;
				return true;
			}
			else if (NumericProperty->IsInteger())
			{
				int value = NumericProperty->GetSignedIntPropertyValue(Value);
				t = (float)value;
				return true;
			}
		}
		return false;
	};

	bool UPropertyToValue(UProperty* Property, FString& t)
	{
		void* Value = Property->ContainerPtrToValuePtr<uint8>(obj);
		if (UStrProperty *StringProperty = Cast<UStrProperty>(Property))
		{
			FString value = StringProperty->GetPropertyValue(Value);
			t = value;
			return true;
		}
		return false;
	}

	bool UPropertyToValue(UProperty* Property, int& t)
	{
		void* Value = Property->ContainerPtrToValuePtr<uint8>(obj);
		if (UNumericProperty *NumericProperty = Cast<UNumericProperty>(Property))
		{
			if (NumericProperty->IsFloatingPoint())
			{
				float value = NumericProperty->GetFloatingPointPropertyValue(Value);
				t = (int)value;
				return true;
			}
			else if (NumericProperty->IsInteger())
			{
				int value = NumericProperty->GetSignedIntPropertyValue(Value);
				t = value;
				return true;
			}
		}
		else if (UEnumProperty* EnumProperty = Cast<UEnumProperty>(Property))
		{
			UEnum* EnumDef = EnumProperty->GetEnum();
			t = EnumProperty->GetUnderlyingProperty()->GetSignedIntPropertyValue(Value);
			return true;
		}
		return false;
	};

	void UpdateDropdownParent()
	{
		for (auto widget : widgetList)
		{
			panel->AddChild(widget);
			if (widget->index == 3)
			{
				UDropdownWidget* dropWidget = (UDropdownWidget*)widget;
				DropdownAttribute* dropAttribut = (DropdownAttribute*)(dropWidget->attribute);
				if (dropAttribut->Parent.IsEmpty())
					continue;
				for (auto widget : widgetList)
				{
					if (widget->index == 3 && widget->memberName == dropAttribut->Parent)
					{
						dropWidget->parent = (UDropdownWidget*)widget;
						dropWidget->InitParentEvent();
					}
				}
			}
		}
	}
};
