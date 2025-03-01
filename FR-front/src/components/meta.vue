<template>
  <div class="app-container" :style="backgroundDiv">
  <el-dialog
    title="对抗样本生成中"
    v-model="dialogTableVisible"
    :show-close="false"
    :center="true"
  >
    <el-progress
      :percentage="100"
      status="success"
      :indeterminate="true"
      :duration="5"
    />
    <span>请耐心等待~</span>
  </el-dialog>
  <el-dialog
    title="测试中"
    v-model="dialogTableVisible2"
    :show-close="false"
    :center="true"
  >
    <el-progress
      :percentage="100"
      status="success"
      :indeterminate="true"
      :duration="5"
    />
    <span>请耐心等待~</span>
  </el-dialog>
  <el-card class="na" style="width: 580px; height: 520px" shadow="never">
    <el-space class="jjjj" direction="vertical" alignment="center" :size="30">
      <el-space class="jj" direction="horizontal" alignment="center" :size="30">
        <el-upload
          class="avatar-uploader"
          ref="upload"
          action="action"
          :show-file-list="false"
          :http-request="uploadFile_1"
        >
          <img v-if="imageUrl_1" :src="imageUrl_1" class="avatar" />
          <el-icon v-else class="avatar-uploader-icon"><Plus />干净样本</el-icon>
        </el-upload>
        <div class="xx" @click="processImage">
          <img v-if="picurl" :src="picurl + '?t=' + Date.now()" class="avatar" />
          <el-icon v-else class="avatar-uploader-icon"><Plus />对抗样本</el-icon>
        </div>
      </el-space>
      <el-space class="jjj" direction="vertical" alignment="center" :size="30">
        <el-table :data="tableData" style="width: 100%" size="medium">
          <!-- <el-table-column prop="score" label="Score" width="180" /> -->
          <el-table-column label="目标模型的识别结果" width="100" align="center">
            <el-table-column prop="score1" label="扰动前" width="180"  align="center">
            </el-table-column>
            <el-table-column prop="score2" label="扰动后" width="180" align="center">
            </el-table-column>
          </el-table-column>
          <el-table-column prop="or" label="攻击成功Yes or No" width="180" align="center" />
        </el-table>
        <el-button
          :icon="SwitchButton"
          @click="getstring"
          style="width: 120px; height: 20px; font-size: 15px"
          >
          测试
        </el-button>
      </el-space>
    </el-space>
    <el-page-header class="but" @back="goBack" />
  </el-card>

  <el-card class="f" style="width: 300px; height: 530px" shadow="never">
    <!-- PGD -->
    <div v-if="this.attack == 'PGD'">
      <div align="left" style="margin: 10px">
        <h3>攻击算法</h3>
      </div>
      <div align="left" style="margin: 10px">
        <el-select
          v-model="attack"
          placeholder="Select attack"
          style="width: 200px"
        >
          <el-option
            v-for="item in attacks"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
      </div>
      <div align="left" style="margin: 10px">
        <h3>目标模型</h3>
      </div>
      <div align="left" style="margin: 10px">
        <el-select
          v-model="metas"
          multiple
          placeholder="Select models"
          style="width: 200px"
        >
          <el-option
            v-for="item in models"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
      </div>
      <div align="left" style="margin: 10px">
        <h3>PGD的参数</h3>
      </div>
      <div align="left" style="margin: 10px">
        <el-input
          :prefix-icon="Edit"
          v-model="eps"
          placeholder="Please input eps"
          clearable
          style="width: 200px"
        />
      </div>
      <div align="left" style="margin: 10px">
        <el-input
          :prefix-icon="Edit"
          v-model="iters"
          placeholder="Please input iters"
          clearable
          style="width: 200px"
        />
      </div>
    </div>
    <!-- FGSM -->
    <div v-if="this.attack == 'FGSM'">
      <div align="left" style="margin: 10px">
        <h3>攻击算法</h3>
      </div>
      <div align="left" style="margin: 10px">
        <el-select
          v-model="attack"
          placeholder="Select attack"
          style="width: 200px"
        >
          <el-option
            v-for="item in attacks"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
      </div>
      <div align="left" style="margin: 10px">
        <h3>目标模型</h3>
      </div>
      <div align="left" style="margin: 10px">
        <el-select
          v-model="metas"
          multiple
          placeholder="Select models"
          style="width: 200px"
        >
          <el-option
            v-for="item in models"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
      </div>
      <div align="left" style="margin: 10px">
        <h3>FGSM参数</h3>
      </div>
      <div align="left" style="margin: 10px">
        <el-input
          :prefix-icon="Edit"
          v-model="eps"
          placeholder="Please input eps"
          clearable
          style="width: 200px"
        />
      </div>
    </div>
    <!-- C&W参数 -->
    <div v-if="this.attack == 'CandW'">
      <div align="left" style="margin: 10px">
        <h3>攻击算法</h3>
      </div>
      <div align="left" style="margin: 10px">
        <el-select
          v-model="attack"
          placeholder="Select attack"
          style="width: 200px"
        >
          <el-option
            v-for="item in attacks"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
      </div>
      <div align="left" style="margin: 10px">
        <h3>目标模型</h3>
      </div>
      <div align="left" style="margin: 10px">
        <el-select
          v-model="metas"
          multiple
          placeholder="Select models"
          style="width: 200px"
        >
          <el-option
            v-for="item in models"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
      </div>
      <div align="left" style="margin: 10px">
        <h3>C&W的参数</h3>
      </div>
      <div align="left" style="margin: 10px">
        <el-input
          :prefix-icon="Edit"
          v-model="iters"
          placeholder="Please input iters"
          clearable
          style="width: 200px"
        />
      </div>
      <div align="left" style="margin: 10px">
        <el-input
          :prefix-icon="Edit"
          v-model="confidence"
          placeholder="Please input confidence"
          clearable
          style="width: 200px"
        />
      </div>
      <div align="left" style="margin: 10px">
        <el-input
          :prefix-icon="Edit"
          v-model="c"
          placeholder="Please input c"
          clearable
          style="width: 200px"
        />
      </div>
    </div>
  </el-card>
  </div>
</template>
<script>
import { Plus } from "@element-plus/icons-vue";
import { Delete } from "@element-plus/icons-vue";
import { SwitchButton } from "@element-plus/icons-vue";
import { Edit } from "@element-plus/icons-vue";
import axios from "axios";
export default {
  name: "app",

  data() {
    return {
      metas: "",
      attack: "PGD",
      attacks: [
        {
          value: "FGSM",
          label: "FGSM",
        },
        {
          value: "CandW",
          label: "C&W",
        },
        {
          value: "PGD",
          label: "PGD",
        },
      ],
      dialogTableVisible: false,
      dialogTableVisible2: false,
      imageUrl_1: "",
      imageUrl_2: "",
      advimg:"",
      Delete,
      SwitchButton,
      Edit,
      picurl: "",
      eps: "",
      goal: "",
      metric: "",
      model: "",
      models: [
        {
          value: "LPRNet",
          label: "LPRNet",
        }
      ],
      
      iters: "",
      confidence:"",
      c:"",
      mu: "",
      length: "",
      kernel_len: "",
      nsig: "",
      tableData: [{ score1: "",score2:"", or: "" }],
      score: "",
      backgroundDiv: {
        backgroundImage: 'url(' + require('@/assets/imagebox/R-C.png') + ')',
        backgroundRepeat: 'no-repeat',
        backgroundSize: 'cover'
      }
    };
  },

  methods: {
    goBack() {
      this.$router.push("/home");
    },
    cc() {
      this.dialogTableVisible = true;
      setTimeout(() => {
        this.dialogTableVisible = false;
      }, 5000);
    },
    // 上传图片
    async uploadFile_1(params) {
      let form = new FormData();
      form.append("file", params.file);

      const res = await axios.post("http://127.0.0.1:5000/upload?src=0", form);
      console.log(res.data);
      this.imageUrl_1 = res.data;
    },
    async uploadFile_2(params) {
      let form = new FormData();
      form.append("file", params.file);

      const res = await axios.post("http://127.0.0.1:5000/upload?src=1", form);
      console.log(res);
      this.imageUrl_2 = res.data;
    },
    //删除图片
    del_1() {
      axios.get("http://127.0.0.1:5000/delete_1", {}).then((res) => {
        this.imageUrl_1 = res.data.img;
      });
    },
    del_2() {
      axios.get("http://127.0.0.1:5000/delete_2", {}).then((res) => {
        this.imageUrl_2 = res.data.img;
      });
    },
    fuju(){
      console.log(this.metas)
    },
    //处理图片
    async processImage() {
      this.dialogTableVisible = true;
      var that = this;
      axios.get(`http://127.0.0.1:5000/process_image?attack=${this.attack}&eps=${this.eps}&iters=${this.iters}
      &confidence=${this.confidence}&c=${this.c}`)
        .then((response) => {
          that.dialogTableVisible = false;
          that.picurl = response.data.imageUrl;
          console.log(that.picurl)
        })
        .catch((error) => {
          console.error('Error loading image:', error);
        });
    },
    //测试样本
    async getstring(){
      this.dialogTableVisible2 = true;
      try {
        const response = await axios.get('http://127.0.0.1:5000/get_string');
        this.tableData = response.data;
        this.dialogTableVisible2 = false;
      } catch (error) {
        console.error('Error fetching string from backend:', error);
        this.dialogTableVisible2 = false;
      }
    }
  },
  components: { Plus },
};
</script>
<style scoped>

.but /deep/ .el-page-header__left::after{
  content: none;
}
.na {
  float: left;
  margin-right: 20px;
  margin-bottom: 20px;
  margin-left: 100px;
  margin-top: 30px;
}
.f {
  float: right;
  margin-right: 100px;
  margin-bottom: 20px;
  margin-left: 20px;
  margin-top: 30px;
}
.xx {
  border: 1px dashed #d9d9d9;
  border-radius: 6px;
}
.avatar-uploader .avatar {
  width: 94px;
  height: 24px;
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.avatar {
  width: 94px;
  height: 24px;
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.jj {
  margin-right: 20px;
  margin-bottom: 20px;
  margin-left: 20px;
  margin-top: 20px;
}
.app-container {
  height: 100%;
  width: 100%;
}
</style>

<style>
.avatar-uploader .el-upload {
  border: 1px dashed #d9d9d9;
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: var(--el-transition-duration-fast);
}

.avatar-uploader .el-upload:hover {
  border-color: var(--el-color-primary);
}

.el-icon.avatar-uploader-icon {
  font-size: 28px;
  color: #8c939d;
  width: 178px;
  height: 178px;
  text-align: center;
}
</style>
