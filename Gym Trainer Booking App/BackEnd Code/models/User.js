import { MoneyOffCsredRounded } from "@mui/icons-material";
const mongoose = require('mongoose');
const userSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true
    },

    email: {
        type: String,
        required: true
    },
    password: {
        type: String,
        required: true
    },
    isAdmin: {
        type: Boolean,
        default:false
    },

    isTrainer: {
        type: Boolean,
        default:false
    }
}, {
    timestamps: true
})
const User = mongoose.models.User || mongoose.model('User', userSchema);

export default User;
