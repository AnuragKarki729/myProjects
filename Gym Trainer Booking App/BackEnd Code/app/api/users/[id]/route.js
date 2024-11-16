import dbConnect from '@/lib/mongodb';
import User from '@/models/User';

export async function GET(req, { params }) {
    await dbConnect();
    const { id } = params; // User ID from the URL

    try {
        const user = await User.findById(id).exec();
        if (!user) {
            return new Response(JSON.stringify({ message: 'User not found' }), { status: 404 });
        }

        return new Response(JSON.stringify(user), { status: 200 });
    } catch (error) {
        return new Response(JSON.stringify({ message: 'Error fetching user', error: error.message }), { status: 500 });
    }
}
