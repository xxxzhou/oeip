using OeipCommon.OeipAttribute;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectBind3D<T> : ObjectBind<T>
{
    private RectTransform panel;

    public override IOeipComponent CreateComponent(ControlAttribute attribute)
    {
        if (attribute is SliderInputAttribute)
        {
            return GameObject.Instantiate(OeipManagerU3D.Instance.SliderInputTemplate, panel);
        }
        else if (attribute is InputAttribute)
        {
            return GameObject.Instantiate(OeipManagerU3D.Instance.InputComponent, panel);
        }
        else if (attribute is ToggleAttribute)
        {
            return GameObject.Instantiate(OeipManagerU3D.Instance.ToggleComponent, panel);
        }
        else if (attribute is DropdownAttribute)
        {
            return GameObject.Instantiate(OeipManagerU3D.Instance.DropdownComponent, panel);
        }
        return null;
    }

    public override bool OnAddPanel<P>(IOeipComponent component, P panel)
    {
        return true;
    }

    public void Bind(T t, RectTransform panel, Action<T> action = null)
    {
        this.panel = panel;
        Bind<RectTransform>(t, panel, action);
    }

    public override void OnBind()
    {
        foreach (var comp in components)
        {
            if (comp is DropdownComponent)
            {
                DropdownComponent dc = comp as DropdownComponent;
                DropdownAttribute da = comp.ControlAttribute as DropdownAttribute;
                if (!string.IsNullOrEmpty(da.Parent))
                {
                    var parent = GetComponent(da.Parent) as DropdownComponent;
                    if (parent != null)
                    {
                        dc.parent = parent;
                    }
                }
            }
        }
    }
}
