import dbConnect from '@/lib/mongodb';
import Trainer from '@/models/Trainer';

const addTrainerEmails = async () => {
  await dbConnect();

  try {
    const trainers = await Trainer.find({ email: { $exists: false } }); // Find trainers without emails

    let trainerCount = 1; // Start the count for trainers

    for (const trainer of trainers) {
      // Assign sequential email addresses
      const email = `trainer${trainerCount}@trainer.com`;

      // Set default password as '123' (In production, you'd want to hash this)
      trainer.email = email;
      trainer.password = '123';

      // Save the updated trainer
      await trainer.save();
      console.log(`Updated trainer ${trainer.name} with email: ${trainer.email}`);

      trainerCount++; // Increment the count for the next trainer
    }

    console.log('All trainers updated with sequential emails and passwords.');
  } catch (error) {
    console.error('Error updating trainers:', error);
  }
};

addTrainerEmails();
