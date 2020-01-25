
using OeipCommon.OeipAttribute;
using UnityEngine.UI;

public class SliderInputComponent : BaseComponent<float>
{
    public Slider slider;
    public InputField field;
    private SliderInputAttribute attribute;
    private float range = 1.0f;
    private bool bAutoRange = false;

    private bool bSliderChange = false;
    private bool bTextChange = false;
    // Use this for initialization
    void Start()
    {
        slider.onValueChanged.AddListener(OnSliderChange);
        field.onValueChanged.AddListener(OnFieldChange);
        text.text = attribute.DisplayName;
    }

    public override void OnSetAttribute()
    {
        attribute = Attribute<SliderInputAttribute>();
        if (attribute.IsAutoRange)
        {
            setValue(attribute.Default, attribute.Range, attribute.IsInt);
        }
        else
        {
            setValue(attribute.Default, attribute.Min, attribute.Max, attribute.IsInt);
        }
    }

    public void OnFieldChange(string value)
    {
        if (bTextChange)
            return;
        bTextChange = true;
        float tv = 0.0f;
        if (float.TryParse(value, out tv))
        {
            if (tv != slider.value)
            {
                OnSetValue(tv);
            }
        }
        bTextChange = false;
    }

    public void OnSliderChange(float value)
    {
        if (bSliderChange)
            return;
        bSliderChange = true;        
        OnValueChange(value);
        bSliderChange = false;
    }

    private void setValue(float defaultValue, float range, bool bInt = false)
    {
        this.range = range;
        this.bAutoRange = true;
        slider.maxValue = defaultValue + range / 2.0f;
        slider.minValue = defaultValue - range / 2.0f;
        slider.value = defaultValue;
        slider.wholeNumbers = bInt;
    }

    private void setValue(float defaultValue, float minValue, float maxValue, bool bInt = false)
    {
        slider.maxValue = maxValue;
        slider.minValue = minValue;
        slider.value = defaultValue;
        slider.wholeNumbers = bInt;
    }

    public override void OnSetValue(float value)
    {
        if (bAutoRange)
        {
            slider.maxValue = value + range / 2.0f;
            slider.minValue = value - range / 2.0f;
        }
        slider.value = value;
        field.text = value.ToString();
    }
}
