import express from 'express';
import mongoose from 'mongoose';       

const app = express();
const port = 3000;

mongoose.connect('mongodb://localhost:27017/myapp', { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.log(err));

  const studentSchema = new mongoose.Schema({
    "name" : {
        type: String,
        required: true
    },
    "age" : {
        type: Number,
        required: true
    }

});

app.get("/", async (req, res) => {
    try {
        const students = await Student.find();
        res.json(students);
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

const Student = mongoose.model('Student', studentSchema);

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});