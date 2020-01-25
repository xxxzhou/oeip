// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Unlit/planMat"
{
	Properties
	{
		_MainTex("Texture", 2D) = "black" {}
	}
		SubShader
	{
		Tags
	{
		"Queue" = "Transparent"		
		"RenderType" = "Transparent"	
	}
		Blend SrcAlpha OneMinusSrcAlpha
		//ZTest Always
		Cull Off
		ZWrite Off
		Lighting Off
		Fog{ Mode Off }
		Pass
	{
		CGPROGRAM
#pragma vertex vert
#pragma fragment frag	

#include "UnityCG.cginc"

		struct appdata
	{
		float4 vertex : POSITION;
		float2 uv : TEXCOORD0;
	};

	struct v2f
	{
		float2 uv : TEXCOORD0;
		float4 vertex : SV_POSITION;
	};

	sampler2D _MainTex;


	v2f vert(appdata v)
	{
		v2f o;
		o.vertex = UnityObjectToClipPos(v.vertex);
		o.uv = v.uv;
		return o;
	}

	fixed4 frag(v2f i) : SV_Target
	{
		fixed4 col = tex2D(_MainTex, i.uv);
		return col;
	}
		ENDCG
	}
	}
}
