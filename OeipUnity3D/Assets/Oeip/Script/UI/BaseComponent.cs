
using OeipCommon.OeipAttribute;
using System;
using UnityEngine;
using UnityEngine.UI;

public class BaseComponent<T> : MonoBehaviour, IOeipComponent, IOeipComponent<T>
{
    public Text text;
    public Action<T, BaseComponent<T>> onValueChangeAction;
    private string displayName = string.Empty;
    private ControlAttribute controlAttribute = null;

    public string DisplayName
    {
        get
        {
            return displayName;
        }
        set
        {
            displayName = name;
            text.text = displayName;
        }
    }

    public ControlAttribute ControlAttribute
    {
        get
        {
            return controlAttribute;
        }
        set
        {
            controlAttribute = value;
            OnSetAttribute();
        }
    }

    public U Attribute<U>() where U : ControlAttribute
    {
        if (ControlAttribute == null)
            return null;
        U t = ControlAttribute as U;
        return t;
    }

    public virtual void OnSetAttribute()
    {

    }

    /// <summary>
    /// 用值去更新UI控件
    /// </summary>
    /// <param name="value"></param>
    public virtual void OnSetValue(T value)
    {
    }

    public void SetValueChangeAction(Action<T, IOeipComponent<T>> onAction)
    {
        onValueChangeAction = onAction;
    }
    /// <summary>
    /// UI对应的值更新
    /// </summary>
    /// <param name="value"></param>
    public void OnValueChange(T value)
    {
        onValueChangeAction?.Invoke(value, this);
    }

    public virtual void UpdateControl(object obj)
    {
        if (ControlAttribute == null)
            return;
        T value = ControlAttribute.GetValue<T>(ref obj);
        OnSetValue(value);
    }
}

