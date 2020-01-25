using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper.FixPipe
{
    /// <summary>
    /// 用于DX11显示所需要的约束
    /// </summary>
    public interface IDXViewPipe
    {
        /// <summary>
        /// GPU计算管线操作类
        /// </summary>
        OeipPipe Pipe { get; }

        /// <summary>
        /// 是否有DX11纹理输出
        /// </summary>
        bool IsGpu { get; }

        /// <summary>
        /// DX11纹理输出索引
        /// </summary>
        int OutGpuIndex { get; }
    }
}
