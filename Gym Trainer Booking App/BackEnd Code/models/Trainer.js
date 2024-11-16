const mongoose = require('mongoose');

// Appointment schema for trainer bookings
const appointmentSchema = new mongoose.Schema({
  date: {
    type: Date,
    required: true,
  },
  duration: {
    type: Number, // Duration in minutes
    default: 60, // Default to 60 minutes (1 hour)
  },
  clientName: {
    type: String,
    required: true,
  }
}, {
  timestamps: true,
});

// Trainer schema
const trainerSchema = mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  age: {
    type: Number,
    required: true
  },
  expYears: {
    type: Number,
    required: true
  },
  phone: {
    type: Number,
    required: true
  },
  price: {
    type: Number,
    required: true
  },
  imgUrl: [],

  // Add email and password fields
  email: {
    type: String,
    required: true,
    unique: true // Ensure each email is unique
  },
  password: {
    type: String,
    required: true
  },

  currentAppointments: [appointmentSchema], // Embed appointments schema
  type: {
    type: String,
    enum: ['Personal', 'Group', 'CrossFit', 'Physical therapy', 'Bodybuilding', 'Athleticism'],
  },
  busyDates: { // New field to store busy dates
    type: [Date],
    default: []
},
  description: {
    type: String,
    required: true
  },
}, {
  timestamps: true
});

const Trainer = mongoose.models.Trainer || mongoose.model('Trainer', trainerSchema);
module.exports = Trainer;
