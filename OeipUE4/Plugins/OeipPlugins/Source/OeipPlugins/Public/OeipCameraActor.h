// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "OeipDisplayActor.h"
#include "OeipCameraActor.generated.h"

UCLASS()
class OEIPPLUGINS_API AOeipCameraActor : public AActor
{
	GENERATED_BODY()

public:
	// Sets default values for this actor's properties
	AOeipCameraActor();
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		AOeipDisplayActor *CameraShow;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		UTexture2D *nullTex;
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;
};
