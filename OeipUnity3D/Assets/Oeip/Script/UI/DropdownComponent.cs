
using OeipCommon.OeipAttribute;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DropdownComponent : BaseComponent<int>
{
    public Dropdown dropdown;
    private DropdownAttribute attribute = null;
    public DropdownComponent parent;

    private Func<List<string>> onFillFunc;
    private Func<int, List<string>> onFillParentFunc;
    private bool bAutoAddDefault = false;
    // Use this for initialization

    void Start()
    {
        dropdown.onValueChanged.AddListener(onDrowdownChange);
        text.text = attribute.DisplayName;
    }

    public void onDrowdownChange(int value)
    {
        int index = value;
        if (bAutoAddDefault)
        {
            index = index - 1;
        }
        OnValueChange(index);
    }

    public override void OnSetAttribute()
    {
        attribute = Attribute<DropdownAttribute>();
        bAutoAddDefault = attribute.IsAutoDefault;
        OnSetValue(attribute.Default);
        DisplayName = attribute.DisplayName;
    }

    public override void OnSetValue(int value)
    {
        if (attribute == null)
            return;
        int index = value;
        if (bAutoAddDefault)
        {
            index = index + 1;
        }
        if (index >= 0 && index < dropdown.options.Count)
        {
            dropdown.value = index;
        }
        dropdown.RefreshShownValue();
    }

    private void fillOption()
    {
        dropdown.options.Clear();
        List<string> options = new List<string>();
        if (bAutoAddDefault)
        {
            options.Add("None");
        }
        options.AddRange(onFillFunc());
        dropdown.AddOptions(options);
    }

    private void fillOption(int value)
    {
        dropdown.options.Clear();
        List<string> options = new List<string>();
        if (bAutoAddDefault)
        {
            options.Add("None");
        }
        int index = value;
        if (parent.bAutoAddDefault)
            index = value - 1;
        options.AddRange(onFillParentFunc(index));
        dropdown.AddOptions(options);
    }

    public void SetFillOptions(bool bAuto, Func<List<string>> onFunc)
    {
        bAutoAddDefault = bAuto;
        onFillFunc = onFunc;
        fillOption();
    }

    public void SetFillOptions(bool bAuto, Func<int, List<string>> onFunc)
    {
        bAutoAddDefault = bAuto;
        onFillParentFunc = onFunc;
        if (parent != null)
            parent.dropdown.onValueChanged.AddListener(fillOption);
    }
}
