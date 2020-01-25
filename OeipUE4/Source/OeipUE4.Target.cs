// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;
using System.Collections.Generic;

public class OeipUE4Target : TargetRules
{
    public OeipUE4Target(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;

        ExtraModuleNames.AddRange(new string[] { "OeipUE4", "OeipPlugins" });


    }
}
