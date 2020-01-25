// Fill out your copyright notice in the Description page of Project Settings.

#include "OeipDisplayActor.h"
#include "Engine/Texture2D.h"
#include "MeshUtilities.h"

// Sets default values
AOeipDisplayActor::AOeipDisplayActor() {
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}
void AOeipDisplayActor::SetTexture(UTexture2D * texture) {
	if (materialDynamic)
		materialDynamic->SetTextureParameterValue("mainTex", texture);
}
// Called when the game starts or when spawned
void AOeipDisplayActor::BeginPlay() {
	Super::BeginPlay();

	materialDynamic = UMaterialInstanceDynamic::Create(material, this);
	actor->GetStaticMeshComponent()->SetMaterial(0, materialDynamic);
}

// Called every frame
void AOeipDisplayActor::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);
}

