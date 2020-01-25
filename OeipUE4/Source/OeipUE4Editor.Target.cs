// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;
using System.Collections.Generic;

public class OeipUE4EditorTarget : TargetRules
{
	public OeipUE4EditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;

		ExtraModuleNames.AddRange( new string[] { "OeipUE4", "OeipPlugins" } );
	}
}
