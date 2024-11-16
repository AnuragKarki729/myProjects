import dbConnect from '@/lib/mongodb';
import Trainer from '@/models/Trainer';

export async function GET(req, { params }) {
    await dbConnect();
    const { id } = params; // Trainer ID from the URL

    try {
        const trainer = await Trainer.findById(id);

        if (!trainer) {
            return new Response(JSON.stringify({ message: 'Trainer not found' }), { status: 404 });
        }

        return new Response(JSON.stringify(trainer.busyDates), { status: 200 });
    } catch (error) {
        return new Response(JSON.stringify({ message: 'Error fetching busy dates', error: error.message }), { status: 500 });
    }
}

export async function PUT(req, { params }) {
    await dbConnect();
    const { id } = params; // Trainer ID from the URL
    const { date } = await req.json(); // Busy date from request body

    try {
        const trainer = await Trainer.findById(id);

        if (!trainer) {
            return new Response(JSON.stringify({ message: 'Trainer not found' }), { status: 404 });
        }

        // Add the new busy date
        trainer.busyDates.push(new Date(date)); // Ensure it's a Date object
        await trainer.save();

        return new Response(JSON.stringify({ message: 'Busy date added successfully', busyDates: trainer.busyDates }), { status: 200 });
    } catch (error) {
        return new Response(JSON.stringify({ message: 'Error adding busy date', error: error.message }), { status: 500 });
    }
}
