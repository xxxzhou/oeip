
using OeipCommon.OeipAttribute;
using UnityEngine.UI;


public class ToggleComponent : BaseComponent<bool>
{
    public Toggle toggle;
    private ToggleAttribute attribute;
    // Use this for initialization
    void Start()
    {
        toggle.onValueChanged.AddListener(OnValueChange);
        text.text = attribute.DisplayName;
    }

    public override void OnSetAttribute()
    {
        attribute = Attribute<ToggleAttribute>();
        OnSetValue(attribute.Default);
    }

    public override void OnSetValue(bool value)
    {
        if (attribute == null)
            return;
        toggle.isOn = value;
    }
}
