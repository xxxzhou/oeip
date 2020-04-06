namespace OeipWrapperTest
{
    partial class MediaForm
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
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.btn_openSrcFile = new System.Windows.Forms.Button();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.displayDx111 = new OeipControl.DisplayDx11();
            this.SuspendLayout();
            // 
            // btn_openSrcFile
            // 
            this.btn_openSrcFile.Location = new System.Drawing.Point(8, 12);
            this.btn_openSrcFile.Name = "btn_openSrcFile";
            this.btn_openSrcFile.Size = new System.Drawing.Size(94, 23);
            this.btn_openSrcFile.TabIndex = 0;
            this.btn_openSrcFile.Text = "打开媒体文件";
            this.btn_openSrcFile.UseVisualStyleBackColor = true;
            this.btn_openSrcFile.Click += new System.EventHandler(this.btn_openSrcFile_Click);
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(109, 12);
            this.textBox1.Name = "textBox1";
            this.textBox1.Size = new System.Drawing.Size(350, 21);
            this.textBox1.TabIndex = 1;
            // 
            // displayDx111
            // 
            this.displayDx111.BackColor = System.Drawing.SystemColors.ButtonFace;
            this.displayDx111.Location = new System.Drawing.Point(12, 41);
            this.displayDx111.Name = "displayDx111";
            this.displayDx111.Size = new System.Drawing.Size(447, 257);
            this.displayDx111.TabIndex = 2;
            // 
            // MediaForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(487, 326);
            this.Controls.Add(this.displayDx111);
            this.Controls.Add(this.textBox1);
            this.Controls.Add(this.btn_openSrcFile);
            this.Name = "MediaForm";
            this.Text = "MediaForm";
            this.Load += new System.EventHandler(this.MediaForm_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.Button btn_openSrcFile;
        private System.Windows.Forms.TextBox textBox1;
        private OeipControl.DisplayDx11 displayDx111;
    }
}