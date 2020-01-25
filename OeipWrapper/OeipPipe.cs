using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper
{
    public enum OeipDataType
    {
        OEIP_CU8C1 = 0,
        OEIP_CU8C3 = 16,
        OEIP_CU8C4 = 24,
    }

    public class OeipPipe
    {
        protected bool bSetInput = false;
        public int PipeId { get; private set; } = -1;
        public OeipGpgpuType GpgpuType { get; private set; }

        protected OnProcessDelegate onProcessDelegate;
        public event OnProcessDelegate OnProcessEvent;

        public OeipPipe()
        {
            onProcessDelegate = new OnProcessDelegate(OnProcessHandle);
        }

        public void SetPipeId(int id)
        {
            this.PipeId = id;
            OeipHelper.setPipeDataAction(PipeId, onProcessDelegate);
            GpgpuType = OeipHelper.getPipeType(PipeId);
        }

        public void Close()
        {
            bSetInput = false;
            OeipHelper.closePipe(this.PipeId);
        }

        public bool IsEmpty()
        {
            return OeipHelper.emptyPipe(PipeId);
        }

        private void OnProcessHandle(int layerIndex, IntPtr data, int width, int height, int outputIndex)
        {
            OnProcessEvent?.Invoke(layerIndex, data, width, height, outputIndex);
        }

        public int AddLayer(string layerName, OeipLayerType layerType)
        {
            return OeipHelper.addPiepLayer(PipeId, layerName, layerType, IntPtr.Zero);
        }

        public int AddLayer<T>(string layerName, OeipLayerType layerType, T t) where T : unmanaged
        {
            int layerIndex = OeipHelper.addPiepLayer(PipeId, layerName, layerType, IntPtr.Zero);
            UpdateParamet(layerIndex, t);
            return layerIndex;
        }

        public unsafe bool UpdateParamet<T>(int layerIndex, T t) where T : unmanaged
        {
            return OeipHelper.updatePipeParamet(PipeId, layerIndex, &t);
        }

        /// <summary>
        /// 有些结构用不了unmanaged,请用这个方法
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="layerIndex"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public unsafe bool UpdateParametStruct<T>(int layerIndex, T t) where T : struct
        {
            int length = Marshal.SizeOf(typeof(T));
            //auto recycle
            byte* numbers = stackalloc byte[length];
            IntPtr ptr = new IntPtr(numbers);
            Marshal.StructureToPtr(t, ptr, false);
            bool bResult = OeipHelper.updatePipeParamet(PipeId, layerIndex, numbers);
            return bResult;
        }

        public int AddLayerStruct<T>(string layerName, OeipLayerType layerType, T t) where T : struct
        {
            int layerIndex = OeipHelper.addPiepLayer(PipeId, layerName, layerType, IntPtr.Zero);
            UpdateParametStruct(layerIndex, t);
            return layerIndex;
        }

        public void ConnectLayer(int layerIndex, string forwardName, int inputIndex = 0, int selfIndex = 0)
        {
            OeipHelper.connectLayerName(PipeId, layerIndex, forwardName, inputIndex, selfIndex);
        }

        public void ConnectLayer(int layerIndex, int forwardIndex, int inputIndex = 0, int selfIndex = 0)
        {
            OeipHelper.connectLayerIndex(PipeId, layerIndex, forwardIndex, inputIndex, selfIndex);
        }

        public void SetEnableLayer(int layerIndex, bool bEnable)
        {
            OeipHelper.setEnableLayer(PipeId, layerIndex, bEnable);
        }

        public void SetEnableLayerList(int layerIndex, bool bEnable)
        {
            OeipHelper.setEnableLayerList(PipeId, layerIndex, bEnable);
        }

        public void SetInput(int layerIndex, int width, int height, OeipDataType dataType = OeipDataType.OEIP_CU8C1, int inputIndex = 0)
        {
            OeipHelper.setPipeInput(PipeId, layerIndex, width, height, (int)dataType, inputIndex);
            bSetInput = true;
        }

        public void UpdatePipeInputGpuTex(int layerIndex, IntPtr device, IntPtr tex, int inputIndex = 0)
        {
            OeipHelper.updatePipeInputGpuTex(PipeId, layerIndex, device, tex, inputIndex);
        }

        public void UpdatePipeOutputGpuTex(int layerIndex, IntPtr device, IntPtr tex, int outputIndex = 0)
        {
            OeipHelper.updatePipeOutputGpuTex(PipeId, layerIndex, device, tex, outputIndex);
        }

        public void UpdateInput(int layerIndex, IntPtr data, int inputIndex = 0)
        {
            OeipHelper.updatePipeInput(PipeId, layerIndex, data, inputIndex);
        }

        public virtual bool RunPipe()
        {
            if (!bSetInput)
                return false;
            return OeipHelper.runPipe(PipeId);
        }
    }
}
