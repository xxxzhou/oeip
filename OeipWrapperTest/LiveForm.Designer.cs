namespace OeipWrapperTest
{
    partial class LiveForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.button1 = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.button2 = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.liveControl1 = new OeipControl.LiveControl();
            this.cameraControl1 = new OeipControl.CameraControl();
            this.blendControl1 = new OeipControl.BlendControl();
            this.SuspendLayout();
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(63, 503);
            this.textBox1.Name = "textBox1";
            this.textBox1.Size = new System.Drawing.Size(100, 21);
            this.textBox1.TabIndex = 1;
            this.textBox1.Text = "oeiplive1";
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(18, 550);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 3;
            this.button1.Text = "登陆房间";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(16, 529);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(41, 12);
            this.label1.TabIndex = 4;
            this.label1.Text = "label1";
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(99, 550);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(75, 23);
            this.button2.TabIndex = 6;
            this.button2.Text = "登出房间";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(16, 506);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(41, 12);
            this.label2.TabIndex = 7;
            this.label2.Text = "房间名";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(169, 506);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(41, 12);
            this.label3.TabIndex = 8;
            this.label3.Text = "用户ID";
            // 
            // textBox2
            // 
            this.textBox2.Location = new System.Drawing.Point(216, 503);
            this.textBox2.Name = "textBox2";
            this.textBox2.Size = new System.Drawing.Size(62, 21);
            this.textBox2.TabIndex = 9;
            this.textBox2.Text = "10";
            // 
            // liveControl1
            // 
            this.liveControl1.Location = new System.Drawing.Point(12, 328);
            this.liveControl1.Name = "liveControl1";
            this.liveControl1.Size = new System.Drawing.Size(232, 158);
            this.liveControl1.TabIndex = 5;
            // 
            // cameraControl1
            // 
            this.cameraControl1.Location = new System.Drawing.Point(12, 12);
            this.cameraControl1.Name = "cameraControl1";
            this.cameraControl1.Size = new System.Drawing.Size(477, 310);
            this.cameraControl1.TabIndex = 0;
            // 
            // blendControl1
            // 
            this.blendControl1.Location = new System.Drawing.Point(251, 329);
            this.blendControl1.Name = "blendControl1";
            this.blendControl1.Size = new System.Drawing.Size(238, 157);
            this.blendControl1.TabIndex = 10;
            // 
            // LiveForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(502, 577);
            this.Controls.Add(this.blendControl1);
            this.Controls.Add(this.textBox2);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.liveControl1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.textBox1);
            this.Controls.Add(this.cameraControl1);
            this.Name = "LiveForm";
            this.Text = "LiveForm";
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.LiveForm_FormClosed);
            this.Load += new System.EventHandler(this.LiveForm_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private OeipControl.CameraControl cameraControl1;
        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Label label1;
        private OeipControl.LiveControl liveControl1;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox textBox2;
        private OeipControl.BlendControl blendControl1;
    }
}