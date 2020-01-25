// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "OeipSetting.h"
#include "OeipHelperLibrary.generated.h"

/**
 *
 */
UCLASS()
class OEIPPLUGINS_API UOeipHelperLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()
public:
	UFUNCTION(BlueprintPure, Category = Oeip)
		static FString FStreamQualityToFString(const FCameraInfo& cameraInfo);

};
