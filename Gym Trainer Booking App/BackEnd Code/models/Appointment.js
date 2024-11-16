const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const appointmentSchema = new Schema({
  trainer: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Trainer',
    required: true
  },
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  date: {
    type: Date,
    required: true
  },
  status: {
    type: String,
    enum: ['Pending', 'Confirmed', 'Cancelled'],
    default: 'Pending'
  },
}, {
  timestamps: true // Automatically adds createdAt and updatedAt fields
});
appointmentSchema.methods.populateNames = async function() {
  await this.populate('trainer', 'name');
  await this.populate('user', 'username');
};

module.exports = mongoose.models.Appointment || mongoose.model('Appointment', appointmentSchema);
