import connect from "@/lib/mongodb";

export async function register() {
  console.log("API Endpoint:", process.env.NEXT_PUBLIC_API_URL)
  console.log("Connecting to database...");
  await connect();
}
