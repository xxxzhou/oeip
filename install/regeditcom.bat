@echo start regedit 
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\regasm ../x64/Release/OeipLiveCom.dll /u
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\regasm ../x64/Release/OeipLiveCom.dll /verbose /tlb /codebase
@echo end
timeout 5
