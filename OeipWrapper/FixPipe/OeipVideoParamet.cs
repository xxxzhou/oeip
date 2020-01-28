using OeipCommon.OeipAttribute;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper.FixPipe
{
    public class OeipVideoParamet
    {
        [Toggle(DisplayName = "GPU计算", Order = -1)]
        public bool bGpuSeed = false;
        [SliderInput(DisplayName = "IterCount", Order = 0, Default = 1, IsInt = true, Min = 1, Max = 10)]
        public int iterCount = 1;
        [SliderInput(DisplayName = "SeedCount", Order = 1, Default = 1000, IsInt = true, Min = 200, Max = 2000)]
        public int seedCount = 1000;
        [SliderInput(DisplayName = "Count", Order = 2, Default = 50, IsInt = true, Min = 1, Max = 500)]
        public int count = 250;
        [SliderInput(DisplayName = "Gamma", Order = 3, Default = 90.0f, Min = 10.0f, Max = 200.0f)]
        public float gamma = 90.0f;
        [SliderInput(DisplayName = "Lambda", Order = 4, Default = 450.0f, Min = 200.0f, Max = 800.0f)]
        public float lambda = 450.0f;

        [SliderInput(DisplayName = "Softness", Order = 6, Default = 5, IsInt = true, Min = 1, Max = 30)]
        public int softness = 5;
        [SliderInput(DisplayName = "EPS", Order = 7, Default = 5, Min = 1.0f, Max = 10.0f)]
        public float epslgn10 = 5.0f;
        [SliderInput(DisplayName = "Intensity", Order = 8, Default = 0.2f, Min = 0.0f, Max = 1.0f)]
        public float intensity = 0.2f;
    }
}
