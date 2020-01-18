using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipCommon.OeipAttribute
{
    public interface IOeipComponent
    {
        /// <summary>
        /// 控件关联的属性,默认设定Set时,调用OnSetAttribute
        /// </summary>
        ControlAttribute ControlAttribute { get; set; }
        /// <summary>
        /// 控件的定义显示
        /// </summary>
        string DisplayName { get; set; }
        /// <summary>
        /// 当设定ControlAttribute时需要去做什么
        /// </summary>
        void OnSetAttribute();
        /// <summary>
        /// 用结构或类更新当前控件的值
        /// </summary>
        /// <param name="obj"></param>
        void UpdateControl(object obj);
    }

    /// <summary>
    /// 与控件相应的值有关
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IOeipComponent<T> : IOeipComponent
    {
        /// <summary>
        /// 设定当控件值改变后的回调
        /// </summary>
        /// <param name="action"></param>
        void SetValueChangeAction(Action<T, IOeipComponent<T>> action);

        /// <summary>
        /// 控件改变后调用
        /// </summary>
        /// <param name="value"></param>
        void OnValueChange(T value);
    }
}
