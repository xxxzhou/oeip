// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class OeipPlugins : ModuleRules
{
    public OeipPlugins(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicIncludePaths.AddRange(
            new string[] {
				// ... add public include paths required here ...
                Path.Combine(ModuleDirectory, "Public"),
                Path.Combine(ModuleDirectory, "UI"),
                Path.Combine(ModuleDirectory, "Oeip"),
            }
            );
        PrivateIncludePaths.AddRange(
            new string[] {
                Path.Combine(ModuleDirectory, "Private")
				// ... add other private include paths required here ...
            }
            );
        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                 "Core","CoreUObject","Projects","Engine","RHI", "RenderCore", "UMG", "SlateCore", "Json","JsonUtilities",
				// ... add other public dependencies that you statically link with here ...
			}
            );
        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "Engine",
				// ... add private dependencies that you statically link with here ...	
			}
            );
        DynamicallyLoadedModuleNames.AddRange(
            new string[]
            {
				// ... add any modules that your module loads dynamically here ...
			}
            );
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            LoadOeip(Target);
        }
    }

    private string ThirdPartyPath
    {
        get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/")); }
    }

    private void CopyToBinaries(string Filepath, ReadOnlyTargetRules Target)
    {
        string projectPath = Directory.GetParent(ModuleDirectory).Parent.Parent.ToString();
        string binariesDir = Path.Combine(projectPath, "Binaries", Target.Platform.ToString());
        string filename = Path.GetFileName(Filepath);

        if (!Directory.Exists(binariesDir))
            Directory.CreateDirectory(binariesDir);

        if (!File.Exists(Path.Combine(binariesDir, filename)))
            File.Copy(Filepath, Path.Combine(binariesDir, filename), true);
    }

    public bool LoadOeip(ReadOnlyTargetRules Target)
    {
        string OeipPath = Path.Combine(ThirdPartyPath, "Oeip");
        string libPath = Path.Combine(OeipPath, "lib");
        string dllPath = Path.Combine(OeipPath, "bin");

        PublicIncludePaths.AddRange(new string[] { Path.Combine(OeipPath, "include") });
        PublicLibraryPaths.Add(libPath);
        PublicAdditionalLibraries.Add("oeip.lib");
        PublicAdditionalLibraries.Add("oeip-live.lib");
        //需要加载的dll,不加的话,需把相应dll拷贝到工程的Binaries,否则编辑器到75%就因加载不了dll crash.     
        PublicDelayLoadDLLs.Add("oeip.dll");
        PublicDelayLoadDLLs.Add("oeip-live.dll");
        foreach (string path in Directory.GetFiles(dllPath))
        {
            RuntimeDependencies.Add(path);
        }
        return true;
    }
}
